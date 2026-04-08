import numpy as np
from control import dlqr
from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper

from .base import KLQR, LQRParams


class StandardStateForm(KLQR):
    def __init__(self, 
                 model_wrapper : DMDcWrapper | EDMDcWrapper | DeepModelWrapper,
                 lqr_params: LQRParams,
                 use_preview: bool = False,
                 logger=None):
        
        # Logging & Identification
        self._logger = logger
        self._use_preview = use_preview

        # Initialize base class to set model and LQR params
        super().__init__(model_wrapper, lqr_params)

        # Initialize preview matrices
        self.K_ff = None
        self.A_cl_T = None
        self.C_T_Q = None
        self.M_ss = None

        # Compute LQR gain and offline matrices
        self.K = self._dare_solution()

    @property
    def params(self) -> LQRParams:
        return self.lqr_params
        
    def _dare_solution(self):
        A = self.model.dyn.A
        B = self.model.dyn.B
        C = self.model.dyn.C
        Qy = self.lqr_params.weights.Q
        Ru = self.lqr_params.weights.R_abs
        self.Qz = C.T @ Qy @ C
        K, P, _ = dlqr(A, B, self.Qz, Ru)

        self.A_cl_T = (A - B @ K).T

        # Analyze Closed-Loop Stability
        eigvals = np.linalg.eigvals(self.A_cl_T)
        eig_magnitudes = np.abs(eigvals)
        max_eig = np.max(eig_magnitudes)

        if self._logger:
            # A system is stable if all eigenvalues lie strictly inside the unit circle
            if max_eig < 1.0 + 1e-10: 
                self._logger.info(f"System is stable (spectral radius: {max_eig:.2e})")
            else:
                self._logger.warning(f"System is unstable (max eig magnitude: {max_eig:.2e})")

        self.K_ff = np.linalg.solve(Ru + B.T @ P @ B, B.T)

        I = np.eye(self.A_cl_T.shape[0])
        self.M_ss = np.linalg.solve(I - self.A_cl_T, self.Qz)

        return K

    def __post_set_params_update(self):
        self.K = self._dare_solution()

    def set_params(self, **kwargs):
        if 'use_preview' in kwargs:
            self._use_preview = kwargs.pop('use_preview')
        super().set_params(**kwargs)
        self.__post_set_params_update()

    def compute_control(self, x, y_ref):
        """
        Compute control input based on the current mode.
        Args:
            x: Current state array (1D)
            y_cmd: Target command. Can be 1D (Setpoint) or 2D array (Trajectory: N_p x n_y)
        """
        if y_ref.ndim == 1:
            y_ref_scaled = self.model.scaler_y.transform(y_ref.reshape(1, -1))
            z_ref_scaled = self.model.lift(y_ref_scaled).reshape(-1,1).flatten()
        else: 
            y_ref_scaled = self.model.scaler_y.transform(y_ref)
            z_ref_scaled = self.model.lift(y_ref_scaled)

        # Lift and Scale State
        x_scaled = self.model.scaler_x.transform(x.reshape(1, -1)).flatten() if self.model.scaler_x else x.flatten()
        z_scaled = self.model.lift(x_scaled)
        
        if self._use_preview:
            s = self.M_ss @ z_ref_scaled[-1]
            for i in range(z_ref_scaled.shape[0] - 2, -1, -1):
                s = self.A_cl_T @ s + self.Qz @ z_ref_scaled[i]
        else:
            s = self.M_ss @ z_ref_scaled

        u_scaled = -self.K @ z_scaled + self.K_ff @ s
        
        if self.model.scaler_u:
            return self.model.scaler_u.inverse_transform(u_scaled.reshape(1, -1)).flatten()
        return u_scaled.flatten()
    

class StandardOutputForm(KLQR):
    def __init__(self, 
                 model_wrapper : DMDcWrapper | EDMDcWrapper | DeepModelWrapper,
                 lqr_params: LQRParams,
                 use_preview: bool = False,
                 logger=None):
        
        # Logging & Identification
        self._logger = logger
        self._use_preview = use_preview

        # Initialize base class to set model and LQR params
        super().__init__(model_wrapper, lqr_params)

        # Initialize preview matrices
        self.K_ff = None
        self.A_cl_T = None
        self.C_T_Q = None
        self.M_ss = None

        # Compute LQR gain and offline matrices
        self.K = self._dare_solution()

    @property
    def params(self):
        return self.lqr_params
        
    def _dare_solution(self):
        A = self.model.dyn.A
        B = self.model.dyn.B
        C = self.model.dyn.C
        Q_y = self.lqr_params.weights.Q
        R_u = self.lqr_params.weights.R_abs

        # 1. Solve DARE for Feedback Gain
        Qz = C.T @ Q_y @ C
        K, P, _ = dlqr(A, B, Qz, R_u)
        
        if self._logger:
            min_eig = np.min(np.real(np.linalg.eigvals(P)))
            if min_eig > -1e-10:
                self._logger.info(f"DARE P is stable (min eig: {min_eig:.2e})")
            else:
                self._logger.warning(f"DARE P has negative eigenvalue: {min_eig:.2e}")

        # 2. Offline Computations for Preview/Setpoint
        H_inv = np.linalg.inv(R_u + B.T @ P @ B)
        self.K_ff = H_inv @ B.T
        self.A_cl_T = (A - B @ K).T
        self.C_T_Q = C.T @ Q_y

        # 3. Steady-State Feedforward Matrix (M_ss)
        I = np.eye(self.A_cl_T.shape[0])
        self.M_ss = np.linalg.inv(I - self.A_cl_T) @ self.C_T_Q

        return K

    def __post_set_params_update(self):
        self.K = self._dare_solution()

    def set_params(self, **kwargs):
        if 'use_preview' in kwargs:
            self._use_preview = kwargs.pop('use_preview')
        super().set_params(**kwargs)
        self.__post_set_params_update()

    def compute_control(self, x, y_cmd):
        """
        Compute control input based on the current mode.
        Args:
            x: Current state array (1D)
            y_cmd: Target command. Can be 1D (Setpoint) or 2D array (Trajectory: N_p x n_y)
        """
        # Lift and Scale State
        x_scaled = self.model.scaler_x.transform(x.reshape(1, -1)).flatten() if self.model.scaler_x else x.flatten()
        z_scaled = self.model.lift(x_scaled)
        
        if self._use_preview:
            y_scaled = self.model.scaler_y.transform(np.array(y_cmd)) if self.model.scaler_y else np.array(y_cmd)
            s = self.M_ss @ y_scaled[-1]

            for i in range(y_scaled.shape[0] - 2, 0, -1):
                s = self.A_cl_T @ s + self.C_T_Q @ y_scaled[i]
        else:
            y_scaled = self.model.scaler_y.transform(np.array(y_cmd).reshape(1, -1)) if self.model.scaler_y else np.array(y_cmd).reshape(1, -1)
            y_ref = y_scaled[0] if y_scaled.ndim == 2 else y_scaled
            s = self.M_ss @ y_ref

        u_scaled = -self.K @ z_scaled + self.K_ff @ s
        
        # 5. Inverse Scale Output to Physical Units
        if self.model.scaler_u:
            return self.model.scaler_u.inverse_transform(u_scaled.reshape(1, -1)).flatten()
        return u_scaled.flatten()