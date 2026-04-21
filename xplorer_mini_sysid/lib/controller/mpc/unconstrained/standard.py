import numpy as np
from control import dlqr
from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper

from ..base import KMPC, MPCParams


class UnconstrainedStateForm(KMPC):
    def __init__(self, 
                 model_wrapper, 
                 mpc_params, 
                 use_preview: bool = False, 
                 logger=None):
        
        self._logger = logger
        self._use_preview = use_preview
        super().__init__(model_wrapper, mpc_params)

        self.N_horizon = self.mpc_params.N_horizon
        self.A = self.model.dyn.A
        self.B = self.model.dyn.B
        self.Qz = None
        self.R = self.mpc_params.weights.R_abs
        self.P_terminal = None 
        
        self._setup_matrices()
        self._logger.info("Unconstrained State Form MPC initialized successfully.")

    @property
    def params(self):
        return self.mpc_params
    
    def _setup_matrices(self):
        """Precompute weights and check theoretical infinite-horizon stability."""
        C = self.model.dyn.C
        Qy = self.mpc_params.weights.Q
        self.Qz = C.T @ Qy @ C
        
        # Solve DARE to get the steady-state P
        K_ss, P_ss, _ = dlqr(self.A, self.B, self.Qz, self.R)
        self.P_terminal = P_ss

        # Theoretical check: Check if (A, B) is even stabilizable
        eig_cl = np.linalg.eigvals(self.A - self.B @ K_ss)
        max_rho = np.max(np.abs(eig_cl))
        
        if self._logger:
            if max_rho < 1.0 + 1e-10:
                self._logger.info(f"Offline Stability Check: PASSED (rho = {max_rho:.4f})")
            else:
                self._logger.error(f"Offline Stability Check: FAILED (rho = {max_rho:.4f}). Check your Koopman Model!")

    def _solve_finite_horizon_recursion(self, z_ref_trajectory):
        """Backward pass with internal stability monitoring."""
        N_p = z_ref_trajectory.shape[0]     
        S = self.P_terminal
        v = self.P_terminal @ z_ref_trajectory[-1].reshape(-1, 1)
        K_k, K_ff_k, v_next = None, None, None

        for i in range(N_p - 1, -1, -1):
            inv_term = np.linalg.inv(self.B.T @ S @ self.B + self.R)
            K_i = inv_term @ self.B.T @ S @ self.A
            K_ff_i = inv_term @ self.B.T
            
            if i == 0:
                K_k = K_i
                K_ff_k = K_ff_i
                v_next = v 
                self._verify_online_stability(K_k)
            
            A_cl_T = (self.A - self.B @ K_i).T
            v = A_cl_T @ v + self.Qz @ z_ref_trajectory[i].reshape(-1, 1)
            S = self.A.T @ S @ (self.A - self.B @ K_i) + self.Qz
            
        return K_k, K_ff_k, v_next

    def _verify_online_stability(self, K_current):
        """Check the spectral radius of the gain being applied to the AUV."""
        A_cl = self.A - self.B @ K_current
        rho = np.max(np.abs(np.linalg.eigvals(A_cl)))
        if rho >= 1.0 + 1e-10:
            if self._logger:
                self._logger.warning(f"Online Stability Warning: rho = {rho:.4f} >= 1.0!")

    def compute_control(self, x, y_ref):
        # 1. Transform Reference (Implicit Mode Selection via Input Dimension)
        if y_ref.ndim == 1:
            # Setpoint Mode: Hold the single value across the predefined horizon (ZOH)
            y_ref_scaled = self.model.scaler_y.transform(y_ref.reshape(1, -1))
            z_ref_single = self.model.lift(y_ref_scaled).reshape(1, -1)
            z_ref_trajectory = np.repeat(z_ref_single, self.N_horizon, axis=0)
        else: 
            # Trajectory Mode: Use the provided prediction window directly (Preview)
            y_ref_scaled = self.model.scaler_y.transform(y_ref)
            z_ref_trajectory = self.model.lift(y_ref_scaled)

        # 2. Lift Current State
        x_scaled = self.model.scaler_x.transform(x.reshape(1, -1)).flatten() if self.model.scaler_x else x.flatten()
        z_scaled = self.model.lift(x_scaled).reshape(-1, 1)
        
        # 3. Finite Horizon Calculation
        K_k, K_ff_k, v_next = self._solve_finite_horizon_recursion(z_ref_trajectory)
        
        # 4. Control Law (Anticipatory Action)
        u_scaled = -K_k @ z_scaled + K_ff_k @ v_next

        # 5. Scaling Output to PWM/Force
        if self.model.scaler_u:
            return self.model.scaler_u.inverse_transform(u_scaled.reshape(1, -1)).flatten()
        return u_scaled.flatten()

class UnconstrainedOutputForm(KMPC):

    def __init__(self, 
                 model_wrapper : DMDcWrapper | EDMDcWrapper | DeepModelWrapper,
                 mpc_params: MPCParams,
                 use_preview: bool = False,
                 logger=None):
        
        # Logging & Identification
        self._logger = logger
        self._use_preview = use_preview

        # Initialize base class to set model and LQR params
        super().__init__(model_wrapper, mpc_params)

        # Initialize preview matrices
        self.K_ff = None
        self.A_cl_T = None
        self.C_T_Q = None
        self.M_ss = None

        # Compute LQR gain and offline matrices
        self.K = self._dare_solution()

    @property
    def params(self):
        return self.mpc_params
        
    def _dare_solution(self):
        A = self.model.dyn.A
        B = self.model.dyn.B
        C = self.model.dyn.C
        Q_y = self.mpc_params.weights.Q
        R_u = self.mpc_params.weights.R_abs

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
    

