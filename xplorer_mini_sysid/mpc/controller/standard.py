import os
import shutil

import numpy as np
import scipy.linalg
from casadi import SX, DM
from control import dlqr
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosOcpDims, AcadosModel, AcadosOcpConstraints, AcadosOcpCost
from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper

from ..base import KMPC, MPCParams

class Standard(KMPC):
    def __init__(self, 
                 model_wrapper : DMDcWrapper | EDMDcWrapper | DeepModelWrapper,
                 mpc_params: MPCParams,
                 node_name: str,
                 logger=None):
        
        # Logging & Identification
        self._node_name = node_name
        self._logger = logger

        # Initialize base class to set model and MPC params
        super().__init__(model_wrapper, mpc_params)

        # Setup ACADOS components
        self._dims = self._setup_acados_dims()
        self._model = self._setup_acados_model()
        self._cost = self._setup_acados_cost()
        self._constraints = self._setup_acados_constraints()
        self._solver = self._setup_acados_solver()

    def _dare_solution(self):
        nz = self.model.dyn.A.shape[0]
        Qz = self.model.dyn.C.T @ self.mpc_params.weights.Q @ self.model.dyn.C
        Rz = self.mpc_params.weights.R_abs

        # --- Tikhonov Regularization ---
        _, P, _ = dlqr(self.model.dyn.A, self.model.dyn.B, Qz, Rz)
        
        # Symmetrize P to mitigate numerical issues
        # P = (P + P.T) / 2.0 

        # check if P is positive definite
        min_eig = np.min(np.real(np.linalg.eigvals(P)))
        if min_eig > -1e-10:
            self._logger.info(f"DARE solution P is stable (min eig: {min_eig:.2e})")
        else:
            self._logger.warning(f"DARE solution P has significant negative eigenvalue: {min_eig:.2e}")

        return P
    
    def _setup_acados_model(self) -> AcadosModel:

        model = AcadosModel()
        model.name = f'kmpc_{self._node_name}'
        nz = self.model.dyn.A.shape[0]
        nu = self.model.dyn.B.shape[1]
        sym_z = SX.sym('z', nz)
        sym_u = SX.sym('u', nu)
        model.x, model.u = sym_z, sym_u
        model.disc_dyn_expr = DM(self.model.dyn.A) @ sym_z + DM(self.model.dyn.B) @ sym_u
        
        return model
    
    def _setup_acados_dims(self) -> AcadosOcpDims:
        dims = AcadosOcpDims()
        nz = self.model.dyn.A.shape[0]
        nu = self.model.dyn.B.shape[1]
        ny = self.model.dyn.C.shape[0]
        ny_total = ny + nu

        dims.nx = nz
        dims.nu = nu
        dims.ny = ny_total
        dims.ny_e = nz

        return dims

    def _setup_acados_cost(self) -> AcadosOcpCost:
        cost = AcadosOcpCost()
        nz, nu = self.model.dyn.A.shape[0], self.model.dyn.B.shape[1]
        ny = self.model.dyn.C.shape[0]
        ny_total = ny + nu

        cost.cost_type = 'LINEAR_LS'
        cost.cost_type_e = 'LINEAR_LS'
        
        # Stage Cost: Penalize [y; u]
        Vx = np.zeros((ny_total, nz))
        Vx[:ny, :] = self.model.dyn.C
        cost.Vx = Vx

        Vu = np.zeros((ny_total, nu))
        Vu[ny:, :] = np.eye(nu)
        cost.Vu = Vu

        cost.W = scipy.linalg.block_diag(self.mpc_params.weights.Q, self.mpc_params.weights.R_abs)
        cost.yref = np.zeros(ny_total)

        # Terminal Cost: z^T * P * z
        P = self._dare_solution()
        cost.Vx_e = np.eye(nz) # ต้องเป็น Identity เพื่อแมป z ไปยัง W_e (P)
        cost.W_e = P
        cost.yref_e = np.zeros(nz)
        
        return cost

    def _setup_acados_constraints(self) -> AcadosOcpConstraints:
        constraints = AcadosOcpConstraints()
        nz, nu = self.model.dyn.A.shape[0], self.model.dyn.B.shape[1]
        ny = self.model.dyn.C.shape[0]

        # Input bounds
        tau_max_sc = self.model.scaler_u.transform(np.array(self.mpc_params.bounds.u_max).reshape(1, -1)).flatten()
        constraints.lbu, constraints.ubu = -tau_max_sc, tau_max_sc
        constraints.idxbu = np.arange(nu)

        # Path bounds (Output constraints)
        v_max_sc = self.model.scaler_y.transform(np.array(self.mpc_params.bounds.y_max).reshape(1, -1)).flatten()
        constraints.C = self.model.dyn.C  # ใช้แค่ (ny x nz)
        constraints.D = np.zeros((ny, nu))
        constraints.lg, constraints.ug = -v_max_sc, v_max_sc
        
        # Initial condition
        constraints.idxbx_0 = np.arange(nz)
        constraints.lbx_0 = constraints.ubx_0 = np.zeros(nz)

        return constraints
    
    def _setup_acados_solver(self) -> AcadosOcpSolver:

        json_file = f'acados_{self._node_name}.json'
        if os.path.exists(json_file): os.remove(json_file)
        if os.path.exists('c_generated_code'): shutil.rmtree('c_generated_code')

        # Create and configure the acados OCP solver
        ocp = AcadosOcp()
        ocp.dims = self._dims
        ocp.model = self._model
        ocp.cost = self._cost
        ocp.constraints = self._constraints

        # --- Options ---
        ocp.solver_options.N_horizon = self.mpc_params.N_horizon
        ocp.solver_options.tf = self.mpc_params.N_horizon * self.mpc_params.dt
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.qp_solver_cond_N = 5 
        ocp.solver_options.print_level = 0
        ocp.solver_options.tol = 1e-4 
        
        return AcadosOcpSolver(ocp, json_file=json_file)

    def __post_set_params_update(self):
        if hasattr(self, '_solver'):
            W = scipy.linalg.block_diag(self.mpc_params.weights.Q, 
                                            self.mpc_params.weights.R_abs)
            P = self._dare_solution()

            for i in range(self.mpc_params.N_horizon):
                self._solver.cost_set(i, "W", W)
            self._solver.cost_set(self.mpc_params.N_horizon, "W", P)

            self._logger.info("MPC parameters updated. Recomputed cost matrices based on new parameters.")

    def set_params(self, **kwargs):
        super().set_params(**kwargs)
        self.__post_set_params_update()

    def compute_control(self, x, y_ref):
        x_scaled = self.model.scaler_x.transform(x.reshape(1, -1))
        z_scaled = self.model.lift(x_scaled)
        y_ref_scaled = self.model.scaler_y.transform(y_ref.reshape(1, -1))
        nu = self.model.dyn.B.shape[1]

        self._solver.set(0, "lbx", z_scaled.flatten())
        self._solver.set(0, "ubx", z_scaled.flatten())

        yref_augmented = np.concatenate([y_ref_scaled.flatten(), np.zeros(nu)])
        for i in range(self.mpc_params.N_horizon):
            self._solver.set(i, "yref", yref_augmented)
        
        # Terminal reference 
        self._solver.set(self.mpc_params.N_horizon, "yref", np.zeros_like(z_scaled.flatten()))

        status = self._solver.solve()
        if status != 0:
            self._logger.warning(f"Acados solver failed with status {status}")

        u = self._solver.get(0, "u").reshape(1,-1)
        return self.model.scaler_u.inverse_transform(u).flatten()