import os
import shutil

import numpy as np
import scipy.linalg
from casadi import SX, DM, vertcat
from control import dlqr
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosOcpDims, AcadosModel, AcadosOcpConstraints, AcadosOcpCost
from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper

from ..base import KMPC, MPCParams


class ConstrainedIntegralStateForm(KMPC):
    def __init__(self, 
                 model_wrapper: DMDcWrapper | EDMDcWrapper | DeepModelWrapper, 
                 mpc_params: MPCParams, 
                 node_name: str,
                 int_limit: np.ndarray,
                 use_preview: bool = False,
                 dt: float = 0.1,
                 logger=None):
        
        self._node_name = node_name
        self._logger = logger
        self._use_preview = use_preview
        self.dt = dt
        
        super().__init__(model_wrapper, mpc_params)

        self.nz = self.model.dyn.A.shape[0]
        self.nu = self.model.dyn.B.shape[1]
        self.ny = self.model.dyn.C.shape[0] 
        self.nx_aug = self.nz + self.ny 
        
        self.int_limit = int_limit.reshape(self.ny, 1)

        self.q = np.zeros((self.ny, 1))

        self._dims = self._setup_acados_dims()
        self._model = self._setup_acados_model()
        self._cost = self._setup_acados_cost()
        self._constraints = self._setup_acados_constraints()
        self._solver = self._setup_acados_solver()

    @property
    def params(self) -> MPCParams:
        return self.mpc_params

    def set_params(self, **kwargs):
        super().set_params(**kwargs)
        self.__post_set_params_update()

    def __post_set_params_update(self):
        if hasattr(self, '_solver'):
            if self._logger:
                self._logger.info("MPC parameters updated. Recomputing augmented cost matrices.")
            
            Qz = self.model.dyn.C.T @ self.mpc_params.weights.Q @ self.model.dyn.C
            Qi = self.mpc_params.weights.Qi 
            R = self.mpc_params.weights.R_abs

            W = scipy.linalg.block_diag(Qz, Qi, R)
            P = self._solve_augmented_dare(Qz, Qi, R)

            for i in range(self.mpc_params.N_horizon):
                self._solver.cost_set(i, "W", W)
                
            self._solver.cost_set(self.mpc_params.N_horizon, "W", P)

    def _setup_acados_model(self) -> AcadosModel:
        model = AcadosModel()
        model.name = f'kmpc_{self._node_name}'
        
        z = SX.sym('z', self.nz)
        q = SX.sym('q', self.ny) 
        u = SX.sym('u', self.nu)
        z_ref = SX.sym('z_ref', self.nz)
        
        model.x = vertcat(z, q)
        model.u = u
        model.p = z_ref 
        
        z_next = DM(self.model.dyn.A) @ z + DM(self.model.dyn.B) @ u
        C_mat = DM(self.model.dyn.C)
        q_next = q + C_mat @ (z - z_ref) * self.dt
        
        model.disc_dyn_expr = vertcat(z_next, q_next)
        return model

    def _setup_acados_dims(self) -> AcadosOcpDims:
        dims = AcadosOcpDims()
        dims.nx = self.nx_aug
        dims.nu = self.nu
        dims.np = self.nz 
        dims.ny = self.nx_aug + self.nu
        dims.ny_e = self.nx_aug
        return dims

    def _setup_acados_cost(self) -> AcadosOcpCost:
        cost = AcadosOcpCost()
        
        Qz = self.model.dyn.C.T @ self.mpc_params.weights.Q @ self.model.dyn.C
        Qi = self.mpc_params.weights.Qi 
        R = self.mpc_params.weights.R_abs
        
        cost.cost_type = 'LINEAR_LS'
        cost.Vx = np.zeros((self.nx_aug + self.nu, self.nx_aug))
        cost.Vx[:self.nx_aug, :self.nx_aug] = np.eye(self.nx_aug)
        
        cost.Vu = np.zeros((self.nx_aug + self.nu, self.nu))
        cost.Vu[self.nx_aug:, :] = np.eye(self.nu)
        
        cost.W = scipy.linalg.block_diag(Qz, Qi, R)

        P = self._solve_augmented_dare(Qz, Qi, R)
        cost.cost_type_e = 'LINEAR_LS'
        cost.Vx_e = np.eye(self.nx_aug)
        cost.W_e = P

        ny_cost = self.nx_aug + self.nu
        ny_e = self.nx_aug
        cost.yref = np.zeros(ny_cost)    
        cost.yref_e = np.zeros(ny_e) 

        return cost

    def _solve_augmented_dare(self, Qz, Qi, R):
        A_aug = np.block([
            [self.model.dyn.A, np.zeros((self.nz, self.ny))],
            [self.model.dyn.C * self.dt, np.eye(self.ny)]
        ])
        B_aug = np.block([
            [self.model.dyn.B],
            [np.zeros((self.ny, self.nu))]
        ])
        Q_aug = scipy.linalg.block_diag(Qz, Qi)
        _, P, _ = dlqr(A_aug, B_aug, Q_aug, R)
        return P

    def _setup_acados_constraints(self) -> AcadosOcpConstraints:
        constraints = AcadosOcpConstraints()
        
        if self.mpc_params.bounds.u_max is not None:
            u_max = self.model.scaler_u.transform(np.array(self.mpc_params.bounds.u_max).reshape(1, -1)).flatten()
            constraints.lbu, constraints.ubu = -u_max, u_max
            constraints.idxbu = np.arange(self.nu)
            if self._logger:
                self._logger.info(f"Applied input bounds (scaled): {u_max}")
        else:
            if self._logger:
                self._logger.info("No input bounds applied.")

        constraints.idxbx_0 = np.arange(self.nx_aug)
        constraints.lbx_0 = constraints.ubx_0 = np.zeros(self.nx_aug)
        
        return constraints

    def _setup_acados_solver(self) -> AcadosOcpSolver:
        ocp = AcadosOcp()
        ocp.parameter_values = np.zeros(self.nz)
        ocp.dims, ocp.model, ocp.cost, ocp.constraints = self._dims, self._model, self._cost, self._constraints
        
        ocp.solver_options.N_horizon = self.mpc_params.N_horizon
        ocp.solver_options.tf = self.mpc_params.N_horizon * self.mpc_params.dt
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_max_iter = 20
        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.qp_solver_cond_N = 5 
        ocp.solver_options.print_level = 0
        ocp.solver_options.tol = 1e-4 

        return AcadosOcpSolver(ocp, json_file=f'acados_int_{self._node_name}.json')

    def compute_control(self, x, y_ref):
        y_ref_scaled_traj = self.model.scaler_y.transform(np.atleast_2d(y_ref))
        if y_ref_scaled_traj.shape[0] == 1:
            y_ref_scaled_traj = np.tile(y_ref_scaled_traj, (self.mpc_params.N_horizon + 1, 1))
        
        z_ref_traj = self.model.lift(y_ref_scaled_traj)
        
        x_scaled = self.model.scaler_x.transform(x.reshape(1, -1)).flatten()
        z_curr = self.model.lift(x_scaled).reshape(-1, 1)

        z_ref_0 = z_ref_traj[0].reshape(-1, 1)
        self.q = self.q + self.model.dyn.C @ (z_curr - z_ref_0) * self.dt
        self.q = np.clip(self.q, -self.int_limit, self.int_limit)
        
        x_init = np.vstack([z_curr, self.q]).flatten()
        self._solver.set(0, "lbx", x_init)
        self._solver.set(0, "ubx", x_init)

        for i in range(self.mpc_params.N_horizon):
            yref = np.concatenate([z_ref_traj[i], np.zeros(self.ny), np.zeros(self.nu)])
            self._solver.set(i, "yref", yref)
            self._solver.set(i, "p", z_ref_traj[i])

        self._solver.set(self.mpc_params.N_horizon, "yref", np.concatenate([z_ref_traj[-1], np.zeros(self.ny)]))

        status = self._solver.solve()
        u_0 = self._solver.get(0, "u")
        
        return self.model.scaler_u.inverse_transform(u_0.reshape(1, -1)).flatten()