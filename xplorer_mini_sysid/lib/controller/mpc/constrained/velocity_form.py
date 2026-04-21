import os
import shutil

import numpy as np
import scipy.linalg
from acados_template import (
    AcadosModel,
    AcadosOcp,
    AcadosOcpDims,
    AcadosOcpConstraints,
    AcadosOcpCost,
    AcadosOcpSolver,
)
from casadi import DM, SX
import scipy.linalg as la
import control as ct
from control import ctrb, dlqr
from kmc.utils.model_wrapper import DeepModelWrapper, DMDcWrapper, EDMDcWrapper

from ..base import KMPC, MPCParams
from ....core.model import LinearModel


class VelocityForm(KMPC):
    def __init__(self, 
                 model_wrapper : DMDcWrapper | EDMDcWrapper | DeepModelWrapper,
                 mpc_params: MPCParams,
                 node_name: str, 
                 logger=None,
                 include_absolute_input: bool = False):

        # Logging & Identification
        self._node_name = node_name
        self._logger = logger
        self._include_absolute_input = include_absolute_input

        # Init with base class to set model and MPC params
        super().__init__(model_wrapper, mpc_params)
        
        # Declare storage for augmented model and solver
        self._is_first_iter = True
        self._z_prev_scaled = np.zeros(self.model.dyn.A.shape[0])
        self._u_prev_scaled = np.zeros(self.model.dyn.B.shape[1])
        
        # Dynamics Augmentation
        self._aug_model = self._augment_dynamics()
        self._dims = self._setup_acados_dims()
        self._cost = self._setup_acados_cost()
        self._model = self._setup_acados_model()
        self._constraints = self._setup_acados_constraints()
        self._solver = self._setup_acados_solver()

    def _dare_solution(self):
        """Solve the discrete-time algebraic Riccati equation for the augmented system."""
        nz = self.model.dyn.A.shape[0]
        nu = self.model.dyn.B.shape[1]

        if self._include_absolute_input:
            Q_aug = scipy.linalg.block_diag(self.model.dyn.C.T @ self.mpc_params.weights.Q @ self.model.dyn.C, 
                                            self.mpc_params.weights.R_abs)
            R_aug = self.mpc_params.weights.R_abs + self.mpc_params.weights.R_rate
            S_aug = np.vstack([np.zeros((nz, nu)), self.mpc_params.weights.R_abs])
            P = scipy.linalg.solve_discrete_are(self._aug_model.A, self._aug_model.B, Q_aug, R_aug, e=None, s=S_aug)
        else:
            Q_aug = scipy.linalg.block_diag(np.zeros((nz, nz)), self.mpc_params.weights.Q)
            R_aug = self.mpc_params.weights.R_rate
            P = scipy.linalg.solve_discrete_are(self._aug_model.A, self._aug_model.B, Q_aug, R_aug, e=None, s=None)
        
        # check if P is positive definite
        if np.all(np.linalg.eigvals(P) > 0):
            self._logger.info("DARE solution P is positive definite.")
        else:
            self._logger.info("DARE solution P is not positive definite!")
        self._logger.info(f"DARE solution P: {P}")

        return P

    def _augment_dynamics(self):
        
        # x_aug = [dz_k; e_k]
        # x_aug_{k+1} = A_aug*x_aug_k + B*du_k
        # e_k     = y_k - C*z_k

        nz = self.model.dyn.A.shape[0]        
        ny = self.model.dyn.C.shape[0]

        # A_aug = [A, 0; C*A, I]
        A = np.block([[self.model.dyn.A,                        np.zeros((nz, ny))], 
                      [self.model.dyn.C @ self.model.dyn.A,     np.eye(ny)]])
        # B_aug = [B; C*B]
        B = np.block([[self.model.dyn.B], 
                      [self.model.dyn.C @ self.model.dyn.B]])
        # C_aug = [0, I] 
        C = np.block([np.zeros((ny, nz)), np.eye(ny)])

        # check controllability of augmented system
        ctrb_matrix = ctrb(A, B)
        if np.linalg.matrix_rank(ctrb_matrix) < A.shape[0]:
            self._logger.info(f"Augmented system is not controllable! Rank: {np.linalg.matrix_rank(ctrb_matrix)}, Required: {A.shape[0]}")
        else :
            self._logger.info(f"Augmented system is controllable. Rank: {np.linalg.matrix_rank(ctrb_matrix)}, Required: {A.shape[0]}")
        return LinearModel(A=A, B=B, C=C)
    
    def _setup_acados_model(self) -> AcadosModel:

        model = AcadosModel()
        model.name = f'kmpc_{self._node_name}'

        nx_aug = self._aug_model.A.shape[0]
        nu = self._aug_model.B.shape[1]
        
        sym_x_aug = SX.sym('x_aug', nx_aug)
        sym_du = SX.sym('du', nu)
        model.x, model.u = sym_x_aug, sym_du
        model.disc_dyn_expr = DM(self._aug_model.A) @ sym_x_aug + DM(self._aug_model.B) @ sym_du

        return model
    
    def _setup_acados_dims(self) -> AcadosOcpDims:

        dims = AcadosOcpDims()
        nz = self.model.dyn.A.shape[0]
        ny = self.model.dyn.C.shape[0]
        nu = self.model.dyn.B.shape[1]
        nx_aug = nz + ny

        if self._include_absolute_input:
            dims.nx = nx_aug            # [dz_k, e_k]
            dims.nu = nu                # du_k
            dims.ny = ny + nu + nu      # [y_k, u_k, du_k]
            dims.ny_e = nx_aug          # Terminal cost on full augmented state [dz_k, e_k]
        else:
            dims.nx = nx_aug           # [dz_k, e_k]
            dims.nu = nu                # du_k
            dims.ny = ny + nu           # [y_k, delta_u_k]
            dims.ny_e = nx_aug          # Terminal cost on full augmented state [dz_k, e_k]

        return dims
    
    def _setup_acados_cost(self) -> AcadosOcpCost:
        
        cost = AcadosOcpCost()
        cost.cost_type = 'LINEAR_LS'
        cost.cost_type_e = 'LINEAR_LS' 
        
        nz = self.model.dyn.A.shape[0]
        nu = self.model.dyn.B.shape[1]
        ny = self.model.dyn.C.shape[0]
        nx_aug = nz + ny
    
        if self._include_absolute_input:
            
            # Vx Setup
            Vx = np.zeros((ny + nu + nu, nx_aug))
            Vx[:ny, :nx_aug] = self._aug_model.C        
            Vx[ny:ny+nu, nz:] = np.eye(nu)           
            cost.Vx = Vx

            # Vu Setup
            Vu = np.zeros((ny + nu + nu, nu))
            Vu[ny:ny+nu, :] = np.eye(nu)             
            Vu[ny+nu:, :] = np.eye(nu)               
            cost.Vu = Vu

            # Weights
            W = scipy.linalg.block_diag(self.mpc_params.weights.Q, 
                                        self.mpc_params.weights.R_abs, 
                                        self.mpc_params.weights.R_rate)
            cost.W = W

            # Terminal Cost P 
            P = self._dare_solution()
            cost.Vx_e = np.eye(nx_aug)
            cost.W_e = P

            # Allocate yref vectors
            cost.yref = np.zeros(ny + nu + nu)   
            cost.yref_e = np.zeros(nx_aug)

        else:

            # Vx Setup: [y; 0] 
            Vx = np.zeros((ny + nu, nx_aug))
            Vx[:ny, :nx_aug] = self._aug_model.C  
            cost.Vx = Vx

            # Vu Setup: [0; delta_u]
            Vu = np.zeros((ny + nu, nu))
            Vu[ny:, :] = np.eye(nu)           
            cost.Vu = Vu

            # Weights: diag(Q, R_rate)
            W = scipy.linalg.block_diag(self.mpc_params.weights.Q, 
                                        self.mpc_params.weights.R_rate)
            cost.W = W

            # Terminal Cost P 
            P = self._dare_solution()
            cost.Vx_e = np.eye(nx_aug)
            cost.W_e = P
            # cost.W_e = np.zeros((nx_aug, nx_aug))

            # Allocate yref vectors
            cost.yref = np.zeros(ny + nu)   
            cost.yref_e = np.zeros(nx_aug)

        return cost
    
    def _setup_acados_constraints(self) -> AcadosOcpConstraints:
        
        constraints = AcadosOcpConstraints()
        nz = self.model.dyn.A.shape[0]
        nu = self.model.dyn.B.shape[1]
        ny = self.model.dyn.C.shape[0]
        nx_aug = nz + ny

        delta_tau_max_sc = np.array(self.mpc_params.bounds.du_max).flatten() / self.model.scaler_u.scale_
        # v_max_sc = self.model.scaler_y.transform(np.array(self.mpc_params.bounds.y_max).reshape(1, -1)).flatten()

        constraints.lbu = -delta_tau_max_sc
        constraints.ubu = delta_tau_max_sc
        constraints.idxbu = np.arange(nu)

        # constraints.C = self._aug_model.C
        # constraints.D = np.zeros((ny, nu))

        # constraints.lg = -v_max_sc
        # constraints.ug = v_max_sc
        
        # initial condition constraints
        constraints.idxbx_0 = np.arange(nx_aug)
        constraints.lbx_0 = np.zeros(nx_aug)
        constraints.ubx_0 = np.zeros(nx_aug)

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
        ocp.solver_options.print_level = 0
        ocp.solver_options.tol = 1e-4 
        
        return AcadosOcpSolver(ocp, json_file=json_file)
    
    def _post_set_params_update(self):
        if hasattr(self, '_solver'):
            nz = self.model.dyn.A.shape[0]
            nu = self.model.dyn.B.shape[1]

            if self._include_absolute_input:
                # Stage Cost Matrix
                W = scipy.linalg.block_diag(self.mpc_params.weights.Q, 
                                            self.mpc_params.weights.R_abs, 
                                            self.mpc_params.weights.R_rate)
                
                P = self._dare_solution()

            else:
                W = scipy.linalg.block_diag(self.mpc_params.weights.Q, 
                                            self.mpc_params.weights.R_rate)
                
                P = self._dare_solution()

            for i in range(self.mpc_params.N_horizon):
                self._solver.cost_set(i, "W", W)
            self._solver.cost_set(self.mpc_params.N_horizon, "W", P)

            self._logger.info("MPC parameters updated. Recomputed cost matrices based on new parameters.")

    def set_params(self, **kwargs):
        super().set_params(**kwargs)
        self._post_set_params_update()

    def compute_control(self, x, y, y_ref):
    
        if self._is_first_iter:
            # Initialize previous lifted state with current lifted state
            x_scaled = self.model.scaler_x.transform(x.reshape(1, -1))
            self._z_prev_scaled = self.model.lift(x_scaled)
            self._is_first_iter = False

        # calculate lifted state and error
        x_scaled = self.model.scaler_x.transform(x.reshape(1, -1))
        z_scaled = self.model.lift(x_scaled)
        dz_scaled = z_scaled - self._z_prev_scaled
        self._z_prev_scaled = z_scaled

        # calculate error 
        y_ref_scaled = self.model.scaler_y.transform(y_ref.reshape(1, -1))
        y_scaled = self.model.scaler_y.transform(y.reshape(1, -1))
        e_scaled = y_scaled - y_ref_scaled

        if self._include_absolute_input:
            yref_vec = np.concatenate([np.zeros_like(e_scaled.flatten()), 
                                       np.zeros_like(self._u_prev_scaled), 
                                       np.zeros_like(self._u_prev_scaled)])  # [e_k, u_{k-1}, du_k]
        else: 
            yref_vec = np.concatenate([np.zeros_like(e_scaled.flatten()), 
                                       np.zeros_like(self._u_prev_scaled)])  # [e_k, du_k]

        x0_t = np.concatenate([dz_scaled.flatten(), e_scaled.flatten()])  # [z_k, e_k]

        # Set values for solver
        self._solver.set(0, "lbx", x0_t)
        self._solver.set(0, "ubx", x0_t)
        for i in range(self.mpc_params.N_horizon):
            self._solver.set(i, "yref", yref_vec)
        self._solver.set(self.mpc_params.N_horizon, "yref", np.zeros_like(x0_t))

        # Solve MPC problem
        status = self._solver.solve()
        delta_u_scaled = self._solver.get(0, "u")
        self._u_prev_scaled += delta_u_scaled

        return self.model.scaler_u.inverse_transform(self._u_prev_scaled.reshape(1, -1)).flatten()