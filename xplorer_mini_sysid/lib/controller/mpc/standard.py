import os
import shutil

import numpy as np
import scipy.linalg
from casadi import SX, DM
from control import dlqr
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosOcpDims, AcadosModel, AcadosOcpConstraints, AcadosOcpCost
from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper

from .base import KMPC, MPCParams


class Standard:
    """
    Standard MPC formulation with state cost and input cost. 
    State is the lifted state z, and the cost penalizes deviation of z from a reference trajectory (lifted from y_ref) and control effort.
    """
    def __init__(self, 
                 mode: str,
                 **kwargs):
        
        if mode == 'state_form':
            self.controller = StandardStateForm(**kwargs)
        elif mode == 'output_form':
            self.controller = StandardOutputForm(**kwargs)
        else:
            raise ValueError(f"Unsupported mode: {mode}. Choose 'state_form' or 'output_form'.")

    @property
    def params(self):
        return self.controller.mpc_params
    
    def set_params(self, **kwargs):
        self.controller.set_params(**kwargs)

    def compute_control(self, x, y_ref):
        return self.controller.compute_control(x, y_ref)


class StandardStateForm(KMPC):
    def __init__(self, 
                 model_wrapper : DMDcWrapper | EDMDcWrapper | DeepModelWrapper,
                 mpc_params: MPCParams,
                 node_name: str,
                 use_preview: bool = False,
                 logger=None):
        
        # Logging & Identification
        self._node_name = node_name
        self._logger = logger
        self._use_preview = use_preview

        # Initialize base class to set model and MPC params
        super().__init__(model_wrapper, mpc_params)

        # Setup ACADOS components
        self._dims = self._setup_acados_dims()
        self._model = self._setup_acados_model()
        self._cost = self._setup_acados_cost()
        self._constraints = self._setup_acados_constraints()
        self._solver = self._setup_acados_solver()

    def _dare_solution(self): 
        Qz = self.model.dyn.C.T @ self.mpc_params.weights.Q @ self.model.dyn.C
        Rz = self.mpc_params.weights.R_abs
        _, P, _ = dlqr(self.model.dyn.A, self.model.dyn.B, Qz, Rz)

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
        
        # Stage cost: [z; u] -> size is nz + nu
        # Terminal cost: [z] -> size is nz
        dims.nx, dims.nu = nz, nu
        dims.ny = nz + nu 
        dims.ny_e = nz
        return dims

    def _setup_acados_cost(self) -> AcadosOcpCost:
        cost = AcadosOcpCost()
        nz, nu = self.model.dyn.A.shape[0], self.model.dyn.B.shape[1]

        cost.cost_type = 'LINEAR_LS'
        cost.cost_type_e = 'LINEAR_LS'
        
        # ||z_k - z_cmd||^2_Q + ||u_k||^2_R
        Vx = np.zeros((nz + nu, nz))
        Vx[:nz, :nz] = np.eye(nz)
        cost.Vx = Vx

        Vu = np.zeros((nz + nu, nu))
        Vu[nz:, :] = np.eye(nu)
        cost.Vu = Vu

        # W = block_diag(Q, R)
        Qz = self.model.dyn.C.T @ self.mpc_params.weights.Q @ self.model.dyn.C
        cost.W = scipy.linalg.block_diag(Qz, self.mpc_params.weights.R_abs)
        
        # Terminal Cost: ||z_N - z_ref,N||^2_P
        P = self._dare_solution()
        cost.Vx_e = np.eye(nz) 
        cost.W_e = P
        
        # initialize reference to zero (can be updated online)
        cost.yref = np.zeros(nz + nu)
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
        constraints.C = self.model.dyn.C 
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
            self._logger.info("MPC parameters updated. Recomputed cost matrices based on new parameters.")
            Qz = self.model.dyn.C.T @ self.mpc_params.weights.Q @ self.model.dyn.C
            W = scipy.linalg.block_diag(Qz, self.mpc_params.weights.R_abs)
            P = self._dare_solution()

            for i in range(self.mpc_params.N_horizon):
                self._solver.cost_set(i, "W", W)
            self._solver.cost_set(self.mpc_params.N_horizon, "W", P)
            self._logger.info(25 * "=")

    def set_params(self, **kwargs):
        super().set_params(**kwargs)
        self.__post_set_params_update()

    def compute_control(self, x, y_ref):
        
        y_ref = np.array(y_ref)
        
        if self._use_preview:
            y_ref_scaled = self.model.scaler_y.transform(y_ref) if self.model.scaler_y else y_ref
        else: 
            y_ref = np.tile(y_ref, (self.mpc_params.N_horizon, 1)) if y_ref.ndim == 1 else y_ref[:self.mpc_params.N_horizon]
            y_ref_scaled = self.model.scaler_y.transform(y_ref) if self.model.scaler_y else y_ref.reshape(1, -1)

        x_scaled = self.model.scaler_x.transform(x.reshape(1, -1))
        z_scaled = self.model.lift(x_scaled)
        z_ref_scaled = self.model.lift(y_ref_scaled)

        self._solver.set(0, "lbx", z_scaled.flatten())
        self._solver.set(0, "ubx", z_scaled.flatten())

        nu = self.model.dyn.B.shape[1]
        zeros_u = np.zeros(nu)

        for i in range(self.mpc_params.N_horizon):
            yref_augmented = np.concatenate([z_ref_scaled[i].flatten(), zeros_u])
            self._solver.set(i, "yref", yref_augmented)
        
        self._solver.set(self.mpc_params.N_horizon, "yref", z_ref_scaled[-1].flatten())

        status = self._solver.solve()
        if status != 0:
            self._logger.warning(f"Solver failed status: {status}")

        u_0 = self._solver.get(0, "u").reshape(1, -1)
        return self.model.scaler_u.inverse_transform(u_0).flatten()
    

class StandardOutputForm(KMPC):
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
        cost.Vx_e = np.eye(nz) 
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
        constraints.C = self.model.dyn.C 
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
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.qp_solver_cond_N = 5 
        ocp.solver_options.print_level = 0
        ocp.solver_options.tol = 1e-4 
        
        return AcadosOcpSolver(ocp, json_file=json_file)

    def __post_set_params_update(self):
        if hasattr(self, '_solver'):
            W = scipy.linalg.block_diag(self.model.dyn.C.T @ self.mpc_params.weights.Q @ self.model.dyn.C, 
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
        y_ref = np.array(y_ref)
        
        if y_ref.ndim == 1:
            y_ref_batch = np.tile(y_ref, (self.mpc_params.N_horizon, 1))
        else:
            if y_ref.shape[0] < self.mpc_params.N_horizon:
                pad_size = self.mpc_params.N_horizon - y_ref.shape[0]
                y_ref_batch = np.vstack([y_ref, np.tile(y_ref[-1], (pad_size, 1))])
            else:
                y_ref_batch = y_ref[:self.mpc_params.N_horizon]

        x_scaled = self.model.scaler_x.transform(x.reshape(1, -1))
        z_current = self.model.lift(x_scaled).flatten()        
        y_ref_scaled = self.model.scaler_y.transform(y_ref_batch)

        self._solver.set(0, "lbx", z_current)
        self._solver.set(0, "ubx", z_current)

        nu = self.model.dyn.B.shape[1]
        zeros_u = np.zeros(nu)

        for i in range(self.mpc_params.N_horizon):
            yref_augmented = np.concatenate([y_ref_scaled[i], zeros_u])
            self._solver.set(i, "yref", yref_augmented)
        
        self._solver.set(self.mpc_params.N_horizon, "yref", y_ref_scaled[-1])

        status = self._solver.solve()
        if status != 0:
            self._logger.warning(f"Solver failed status: {status}")

        u_0 = self._solver.get(0, "u").reshape(1, -1)
        return self.model.scaler_u.inverse_transform(u_0).flatten()


