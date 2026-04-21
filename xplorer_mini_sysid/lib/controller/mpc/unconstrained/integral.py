import numpy as np
from control import dlqr
from ..base import KMPC, MPCParams

class UnconstrainedIntegralStateForm(KMPC):
    def __init__(self, 
                 model_wrapper, 
                 mpc_params: MPCParams, 
                 logger=None, 
                 use_preview=False,
                 dt: float = 0.1):
        """
        Unconstrained MPC with State-based Integral Action.
        Integral definition: q_{k+1} = q_k + (z_k - z_ref_k) * dt
        """
        self._logger = logger
        self._use_preview = use_preview
        self.dt = dt 
        
        super().__init__(model_wrapper, mpc_params)

        self.N_horizon = self.mpc_params.N_horizon
        self.A = self.model.dyn.A
        self.B = self.model.dyn.B
        self.C = self.model.dyn.C 
        self.R = self.mpc_params.weights.R_abs
        
        # State dimensions
        self.n_z = self.A.shape[0]  # Dimension of lifted state
        self.n_y = self.C.shape[0]
        
        # Initialize integral state vector based on state dimension (n_z)
        self.q = np.zeros((self.n_z, 1)) 
        
        self._setup_matrices()
        if self._logger:
            self._logger.info("LQI State-Integral MPC initialized successfully.")

    def _setup_matrices(self):
        """Precompute Augmented Weights with State-Integral Dynamics."""
        Qy = self.mpc_params.weights.Q
        self.Qz = self.C.T @ Qy @ self.C
        
        try:
            # Qi must now be (n_z, n_z) to match the state dimension
            Q_I = getattr(self.mpc_params.weights, 'Qi')
            Qi_aug = self.C.T @ Q_I @ self.C

        except AttributeError:
            raise ValueError("Integral MPC requires 'Qi' weights (state-dimension) in mpc_params.weights.")

        # Augmented State-Space: [z; q]
        # z_k+1 = A*z_k + B*u_k
        # q_k+1 = q_k + dt*z_k - dt*z_ref_k
        
        # A_aug = [[A, 0], [I*dt, I]]
        self.A_aug = np.block([
            [self.A,                 np.zeros((self.n_z, self.n_z))],
            [np.eye(self.n_z) * self.dt,  np.eye(self.n_z)]
        ])
        
        # B_aug = [[B], [0]]
        self.B_aug = np.block([
            [self.B],
            [np.zeros((self.n_z, self.B.shape[1]))]
        ])
        
        # Q_aug = [[Qz, 0], [0, Qi]]
        self.Q_aug = np.block([
            [self.Qz,                        np.zeros((self.n_z, self.n_z))],
            [np.zeros((self.n_z, self.n_z)),      Qi_aug]
        ])
        
        # Solve for Steady-State Algebraic Riccati Equation (LQR)
        try:
            K_ss, P_ss, _ = dlqr(self.A_aug, self.B_aug, self.Q_aug, self.R)
            self.P_terminal = P_ss
        except Exception as e:
            if self._logger:
                self._logger.error(f"LQR Solve Failed: {e}")
            raise e

        # Pre-flight Stability Check
        eig_cl = np.linalg.eigvals(self.A_aug - self.B_aug @ K_ss)
        max_rho = np.max(np.abs(eig_cl))
        
        if self._logger:
            if max_rho < 1.0 + 1e-10:
                self._logger.info(f"Closed-Loop Check: PASSED (rho = {max_rho:.4f})")
            else:
                self._logger.warning(f"Closed-Loop Check: WARNING (rho = {max_rho:.4f})")

    def _solve_finite_horizon_recursion(self, z_ref_trajectory, y_ref_scaled_trajectory):
        """Backward pass with State-Integral and Reference Tracking."""
        N_p = z_ref_trajectory.shape[0]     
        S = self.P_terminal
        
        # Terminal Reference (r_N) for Augmented State: [z_ref; 0]
        r_N = np.vstack([z_ref_trajectory[-1].reshape(-1, 1), np.zeros((self.n_z, 1))])
        v = self.P_terminal @ r_N
        
        K_k, K_ff_k, v_next = None, None, None

        for i in range(N_p - 1, -1, -1):
            inv_term = np.linalg.inv(self.B_aug.T @ S @ self.B_aug + self.R)
            K_i = inv_term @ self.B_aug.T @ S @ self.A_aug
            K_ff_i = inv_term @ self.B_aug.T
            
            # Affine term d_i from State-Integral dynamics: [0; -z_ref * dt]
            d_i = np.vstack([
                np.zeros((self.n_z, 1)), 
                -z_ref_trajectory[i].reshape(-1, 1) * self.dt
            ])
            
            v_prime = v - S @ d_i
            
            if i == 0:
                K_k = K_i
                K_ff_k = K_ff_i
                v_next = v_prime
                self._verify_online_stability(K_k)
            
            # Update v and S
            A_cl_T = (self.A_aug - self.B_aug @ K_i).T
            r_i = np.vstack([z_ref_trajectory[i].reshape(-1, 1), np.zeros((self.n_z, 1))])
            
            v = A_cl_T @ v_prime + self.Q_aug @ r_i
            S = self.A_aug.T @ S @ (self.A_aug - self.B_aug @ K_i) + self.Q_aug
            
        return K_k, K_ff_k, v_next

    def _verify_online_stability(self, K_current):
        A_cl = self.A_aug - self.B_aug @ K_current
        rho = np.max(np.abs(np.linalg.eigvals(A_cl)))
        if rho >= 1.0 + 1e-10 and self._logger:
            self._logger.warning(f"LQI Online Stability Warning: rho = {rho:.4f}")

    def compute_control(self, x, y_ref):
        # 1. Transform Reference
        if y_ref.ndim == 1:
            y_ref_scaled = self.model.scaler_y.transform(y_ref.reshape(1, -1))
            z_ref_single = self.model.lift(y_ref_scaled).reshape(1, -1)
            y_ref_scaled_traj = np.repeat(y_ref_scaled, self.N_horizon, axis=0)
            z_ref_trajectory = np.repeat(z_ref_single, self.N_horizon, axis=0)
        else: 
            y_ref_scaled_traj = self.model.scaler_y.transform(y_ref)
            z_ref_trajectory = self.model.lift(y_ref_scaled_traj)

        # 2. Lift Current State
        x_scaled = self.model.scaler_x.transform(x.reshape(1, -1)).flatten() if self.model.scaler_x else x.flatten()
        z_scaled = self.model.lift(x_scaled).reshape(-1, 1)
        
        # 3. Update State Integral: q = q + (z - z_ref) * dt
        z_ref_k = z_ref_trajectory[0].reshape(-1, 1)
        e_z = z_scaled - z_ref_k
        self.q = self.q + (e_z * self.dt)
        
        # Construct Augmented State Vector [z; q]
        z_aug = np.vstack([z_scaled, self.q])
        
        # 4. Finite Horizon Calculation
        K_k, K_ff_k, v_next = self._solve_finite_horizon_recursion(z_ref_trajectory, y_ref_scaled_traj)
        
        # 5. Control Law
        u_scaled = -K_k @ z_aug + K_ff_k @ v_next

        # 6. Inverse Scaling
        if self.model.scaler_u:
            return self.model.scaler_u.inverse_transform(u_scaled.reshape(1, -1)).flatten()
        return u_scaled.flatten()
    
    def reset_integral(self):
        self.q = np.zeros((self.n_z, 1))