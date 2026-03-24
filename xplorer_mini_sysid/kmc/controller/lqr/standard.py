import numpy as np
from control import dlqr
from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper

from .base import KLQR, LQRParams

class Standard(KLQR):
    def __init__(self, 
                 model_wrapper : DMDcWrapper | EDMDcWrapper | DeepModelWrapper,
                 lqr_params: LQRParams,
                 node_name: str,
                 logger=None):
        
        # Logging & Identification
        self._node_name = node_name
        self._logger = logger

        # Initialize base class to set model and LQR params
        super().__init__(model_wrapper, lqr_params)

        # Compute LQR gain matrix K
        self.K = self._dare_solution()

    def _dare_solution(self):
        Qz = self.model.dyn.C.T @ self.lqr_params.weights.Q @ self.model.dyn.C
        Rz = self.lqr_params.weights.R_abs
        K, P, _ = dlqr(self.model.dyn.A, self.model.dyn.B, Qz, Rz)
        
        # Symmetrize P to mitigate numerical issues
        # P = (P + P.T) / 2.0 

        # check if P is positive definite
        min_eig = np.min(np.real(np.linalg.eigvals(P)))
        if min_eig > -1e-10:
            self._logger.info(f"DARE solution P is stable (min eig: {min_eig:.2e})")
        else:
            self._logger.warning(f"DARE solution P has significant negative eigenvalue: {min_eig:.2e}")

        return K

    def __post_set_params_update(self):
        self.K = self._dare_solution()

    def set_params(self, **kwargs):
        super().set_params(**kwargs)
        self.__post_set_params_update()

    def compute_control(self, x, y_ref, y_cmd=None):
        x_scaled = self.model.scaler_x.transform(x.reshape(1, -1)).flatten() if self.model.scaler_x else x
        y_ref_scaled = self.model.scaler_y.transform(y_ref.reshape(1, -1)).flatten() if self.model.scaler_y else y_ref
        z = self.model.lift(x_scaled)
        # u_scaled = -self.K @ (z - self.model.dyn.C.T @ y_ref_scaled) # wrong - should be function of s_{k+1}
        u_scaled = -self.K @ z + y_cmd
        return self.model.scaler_u.inverse_transform(u_scaled.reshape(1, -1)).flatten()