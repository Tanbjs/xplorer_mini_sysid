#!/usr/bin/env python3
import logging
import warnings
import os
import time
import shutil

import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import WrenchStamped, TwistStamped
from xplorer_mini_common_interfaces.srv import SetModel
from xplorer_mini_common_interfaces.msg import AuvStatus

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import torch
import scipy.linalg
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import SX, DM
from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper

class CascadeKoopmanControl(Node):
    def __init__(self):
        super().__init__('gnc_control_cascade_kmpc')
        self.get_logger().info('Cascade KMPC Node has been started.')

        # --- 1. System Config ---
        self.dt = 0.1
        self.mlflow_uri = "https://mlflow.amarr.tan" 
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.mlflow_client = MlflowClient(tracking_uri=self.mlflow_uri)

        # --- 2. State Variables ---
        self.eta = np.zeros(6)
        self.eta_error = np.zeros(6)
        self.vel = np.zeros(6)
        self.m_rb = np.zeros(6)  # Placeholder for mass parameters, not used in control but can be logged
        
        # --- 3. Parameter Declaration ---
        self.declare_parameters(namespace='', parameters=[
            ('model_name', 'dmdc'),
            ('rigid_body_mass', np.zeros(6).tolist()),  # Placeholder, not used in control but can be logged
            ('Kp_pos', [1.0] * 6),
            ('Ki_pos', [0.0] * 6),
            ('Kd_pos', [0.0] * 6),
            ('integral_limit', [5.0] * 6),
            ('max_vel', [1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
            ('rho_penalty', 100.0),
            ('N_horizon', 10),
            ('Q_diag', [20.0, 20.0, 20.0, 10.0, 10.0, 10.0]),
            ('R_diag', [0.1] * 6),
            ('max_tau', [200.0] * 6),
            ('max_delta_tau', [50.0] * 6)
        ])

        self.add_on_set_parameters_callback(self.parameters_callback)

        # --- 4. Controllers Init ---
        self.position_controller = PositionController(
            kp=np.array(self.get_parameter('Kp_pos').value),
            ki=np.array(self.get_parameter('Ki_pos').value),
            kd=np.array(self.get_parameter('Kd_pos').value),
            int_limit=np.array(self.get_parameter('integral_limit').value),
            max_vel=np.array(self.get_parameter('max_vel').value),
            logger=self.get_logger()
        )
        
        self.velocity_controller = VelocityController(
            dt=self.dt, 
            Q_diag=np.array(self.get_parameter('Q_diag').value), 
            R_diag=np.array(self.get_parameter('R_diag').value), 
            max_vel=np.array(self.get_parameter('max_vel').value),
            max_tau=np.array(self.get_parameter('max_tau').value),
            max_delta_tau=np.array(self.get_parameter('max_delta_tau').value),
            rho_penalty=self.get_parameter('rho_penalty').value,
            N_horizon=self.get_parameter('N_horizon').value,
            node_name=self.get_name(),
            logger=self.get_logger()
        )
        
        # Initial Model Load
        self.load_model_and_setup_mpc()

        # --- 5. ROS Interfaces ---
        self.create_subscription(AuvStatus, 'gnc/control_sync', self.odometry_callback, qos_profile_sensor_data)
        self.create_service(SetModel, 'gnc/koopman/set_model', self.set_model_callback)
        self.wrench_pub = self.create_publisher(WrenchStamped, 'gnc/cmd_wrench/wrench_desired', 10)
        self.twist_pub = self.create_publisher(TwistStamped, 'gnc/sysid/twist_desired', 10)
        self.create_timer(self.dt, self.timer_callback)
    
    def parameters_callback(self, params):
        # ... (Parameter update logic remains mostly the same) ...
        for param in params:
            if param.name == 'Kp_pos':
                self.position_controller.kp = np.array(param.value)
                self.get_logger().info(f"Updated Kp_pos: {self.position_controller.kp}")
            elif param.name == 'Ki_pos':
                self.position_controller.ki = np.array(param.value)
                self.get_logger().info(f"Updated Ki_pos: {self.position_controller.ki}")
            elif param.name == 'Kd_pos':
                self.position_controller.kd = np.array(param.value)
                self.get_logger().info(f"Updated Kd_pos: {self.position_controller.kd}")
            elif param.name == 'integral_limit':
                self.position_controller.int_limit = np.array(param.value)
                self.get_logger().info(f"Updated integral limits: {self.position_controller.int_limit}")
            elif param.name == 'max_vel':
                self.position_controller.max_vel = np.array(param.value)
                self.get_logger().info(f"Updated max velocity limits: {self.position_controller.max_vel}")
            elif param.name == 'Q_diag':
                self.velocity_controller.Q_diag = np.array(param.value)
                self.get_logger().info(f"Updated Q_diag: {self.velocity_controller.Q_diag}")
                self.velocity_controller.update_weights(Q_diag=self.velocity_controller.Q_diag) # Update weights online
            elif param.name == 'R_diag':
                self.velocity_controller.R_diag = np.array(param.value)
                self.get_logger().info(f"Updated R_diag: {self.velocity_controller.R_diag}")
                self.velocity_controller.update_weights(R_diag=self.velocity_controller.R_diag) # Update weights online
            elif param.name == 'rho_penalty':
                self.velocity_controller.rho_penalty = param.value
                self.get_logger().info(f"Updated rho_penalty: {self.velocity_controller.rho_penalty}")
                self.velocity_controller._init_acados_solver() # Rebuild solver with new horizon
            elif param.name == 'N_horizon':
                self.velocity_controller.N = param.value
                self.velocity_controller._init_acados_solver() # Rebuild solver with new horizon
            elif param.name == 'max_tau':
                self.velocity_controller.max_tau = np.array(param.value)
                self.get_logger().info(f"Updated max control input limits: {self.velocity_controller.max_tau}")
                self.velocity_controller._init_acados_solver() # Rebuild solver with new constraints
            elif param.name == 'max_delta_tau':
                self.velocity_controller.max_delta_tau = np.array(param.value)
                self.get_logger().info(f"Updated max change in control input limits: {self.velocity_controller.max_delta_tau}")
                self.velocity_controller._init_acados_solver() # Rebuild solver with new constraints
            elif param.name == "rigid_body_mass":
                self.get_logger().info(f"Received new mass parameters: {param.value}")
                self.m_rb = np.array(param.value)

        return SetParametersResult(successful=True)
    
    def odometry_callback(self, msg):
        self.eta = np.array(msg.eta)
        self.eta_error = np.array(msg.eta_e) 
        self.vel = np.array(msg.nu)
    
    def timer_callback(self):
        t_start = time.time()

        if not self.velocity_controller.is_ready:
            return

        # NaN/Inf Guard
        if np.any(np.isnan(self.vel)) or np.any(np.isinf(self.vel)):
            self.get_logger().warn("NaN detected in velocity inputs. Skipping control cycle.")
            self.publish_wrench(np.zeros(6))
            self.publish_twist(self.twist_pub, np.zeros(6))
            return

        # 1. Outer Loop
        v_ref = self.position_controller.compute_control(-self.eta_error, self.dt)
        J,_,_ = self.eulerang_nwu(self.eta[3], self.eta[4], self.eta[5])
        v_ref_body = np.linalg.inv(J) @ v_ref  # Tranvsform to body frame

        # 2. Inner Loop
        tau_cmd = self.velocity_controller.compute_control(self.vel, v_ref_body)
        tau_cmd = np.asarray(tau_cmd).flatten()
        tau_cmd = np.clip(tau_cmd, -200, 200)

        # 3. Publish
        self.publish_wrench(tau_cmd)
        self.publish_twist(self.twist_pub, v_ref_body)

        t_exec = time.time() - t_start
        if t_exec > self.dt:
            self.get_logger().warn(f"Control loop overload! Exec time: {t_exec:.4f}s > {self.dt}s")

    def load_model_and_setup_mpc(self):
        name = self.get_parameter('model_name').value
        try:
            versions = self.mlflow_client.get_latest_versions(name, stages=None)
            if not versions:
                self.get_logger().error(f"No versions found for model '{name}'")
                return False
            
            latest_version = versions[0].version
            self.get_logger().info(f"Loading MLflow model: {name} (Version {latest_version})")
            model_uri = f"models:/{name}/{latest_version}"
            
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            wrapper = loaded_model.unwrap_python_model() 
            
            self.velocity_controller.setup_model(wrapper)
            self.get_logger().info("MPC Controller Initialized successfully.")
            return True

        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            return False

    def set_model_callback(self, request, response):
        # Update parameter first
        param = rclpy.parameter.Parameter('model_name', rclpy.Parameter.Type.STRING, request.model_type)
        self.set_parameters([param])
        
        success = self.load_model_and_setup_mpc()
        response.success = success
        response.message = "Model updated (Latest Version)" if success else "Update failed"
        return response

    def publish_wrench(self, u):
        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = float(u[0]), float(u[1]), float(u[2])
        msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = float(u[3]), float(u[4]), float(u[5])
        self.wrench_pub.publish(msg)

    def publish_twist(self, publisher, nu):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = float(nu[0]), float(nu[1]), float(nu[2])
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = float(nu[3]), float(nu[4]), float(nu[5])
        publisher.publish(msg)

    def eulerang_nwu(self, phi, theta, psi):
        """
        J matrix for NWU (North-West-Up)
        phi: roll, theta: pitch, psi: yaw
        """
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_th,  s_th  = np.cos(theta), np.sin(theta)
        c_ps,  s_ps  = np.cos(psi), np.sin(psi)

        # Rotation matrix R (NWU)
        R = np.array([
            [c_ps*c_th, -s_ps*c_phi + c_ps*s_th*s_phi,  s_ps*s_phi + c_ps*c_phi*s_th],
            [s_ps*c_th,  c_ps*c_phi + s_phi*s_th*s_ps, -c_ps*s_phi + s_th*s_ps*c_phi],
            [-s_th,      c_th*s_phi,                   c_th*c_phi]
        ])
        
        # Transformation matrix T (for angular velocities)
        T = np.array([
            [1,  0,       -s_th],
            [0,  c_phi,    c_th*s_phi],
            [0, -s_phi,    c_th*c_phi]
        ])
        
        J = np.zeros((6, 6))
        J[0:3, 0:3] = R
        J[3:6, 3:6] = T
        
        return J, R, T


class PositionController:
    def __init__(self, kp, ki, kd, int_limit, max_vel, logger=None):
        self.kp = kp; self.ki = ki; self.kd = kd; self.int_limit = int_limit
        self.integral = np.zeros(6)
        self.prev_error = np.zeros(6)
        self.max_vel = np.array(max_vel) 

    def compute_control(self, error, dt):
        if dt <= 0: return np.zeros(6)
        
        derivative = (error - self.prev_error) / dt
        v_p = self.kp * error
        v_d = self.kd * derivative
        v_unsat = v_p + (self.ki * self.integral) + v_d

        # Saturation
        v_cmd = np.clip(v_unsat, -self.max_vel, self.max_vel)
        
        # Anti-windup: Only integrate if we're not saturated in the direction of the error
        is_saturated = (v_unsat > self.max_vel) | (v_unsat < -self.max_vel)
        same_direction = np.sign(error) == np.sign(v_unsat)
        stop_integrating = is_saturated & same_direction
        self.integral += (~stop_integrating) * (error * dt)
        self.integral = np.clip(self.integral, -self.int_limit, self.int_limit)

        self.prev_error = error

        return v_cmd


class VelocityController:
    def __init__(self, dt, Q_diag, R_diag, max_vel, max_tau, max_delta_tau, rho_penalty, node_name, N_horizon=20, logger=None):
        self.dt = dt
        self.N = N_horizon
        self.node_name = node_name
        self.logger = logger
        self.Q = np.diag(np.array(Q_diag))
        self.R = np.diag(np.array(R_diag))
        self.solver = None
        self.is_ready = False
        self.u_prev_scaled = None
        self.max_vel = np.array(max_vel)
        self.max_tau = np.array(max_tau)
        self.max_delta_tau = np.array(max_delta_tau)

    def setup_model(self, wrapper):
        try:
            # 1. Load Model Matrices
            self.A = np.array(wrapper.A, dtype=np.float64)
            self.B = np.array(wrapper.B, dtype=np.float64)
            self.C = np.array(wrapper.C, dtype=np.float64)

            # 2. Setup Scalers & Lift Function 
            self.scaler_x = getattr(wrapper, "scaler_x", None)
            self.scaler_y = getattr(wrapper, "scaler_y", None)
            self.scaler_u = getattr(wrapper, "scaler_u", None)

            if isinstance(wrapper, DeepModelWrapper):
                self.lift_func = lambda x: np.concatenate([
                    x, 
                    wrapper.model.model.encoder(torch.tensor(x).float()).detach().numpy().flatten()
                ])
            elif isinstance(wrapper, EDMDcWrapper):
                self.lift_func = lambda x: wrapper.model._obs_func.transform(x.reshape(1, -1)).flatten()
            elif isinstance(wrapper, DMDcWrapper):
                self.lift_func = lambda x: x  

            # 3. Init Control Memory
            nu = self.B.shape[1]
            self.u_prev_scaled = np.zeros(nu)

            # 4. Cleanup & Build Solver
            self._init_acados_solver()
            self.is_ready = True
            
            if self.logger: self.logger.info(f"Simple MPC initialized for: {self.node_name}")

        except Exception as e:
            if self.logger: self.logger.error(f"Setup failed: {e}")
            raise e

    def _init_acados_solver(self):

        # --- 0. Clean old files before creating new solver ---
        json_file = f'acados_{self.node_name}.json'
        if os.path.exists(json_file): os.remove(json_file)
        if os.path.exists('c_generated_code'): shutil.rmtree('c_generated_code')

        # --- 1. Define Dimensions ---
        nz = self.A.shape[0]
        nu = self.B.shape[1]
        ny = self.C.shape[0]
        nx_aug = nz + nu
        ocp = AcadosOcp()
        ocp.model.name = f'kmpc_{self.node_name}'
        ocp.dims.N = self.N

        # --- 2. Model Dynamics (Linear Augmented) ---
        sym_x = SX.sym('x', nx_aug)
        sym_u = SX.sym('u', nu)
        ocp.model.x, ocp.model.u = sym_x, sym_u

        # A_bar = [A B; 0 I], B_bar = [B; I]
        A_bar = np.block([[self.A, self.B], [np.zeros((nu, nz)), np.eye(nu)]])
        B_bar = np.block([[self.B], [np.eye(nu)]])
        
        # Dynamics: x_{k+1} = A_bar * x_k + B_bar * u_k
        ocp.model.disc_dyn_expr = DM(A_bar) @ sym_x + DM(B_bar) @ sym_u

        # --- 3. Cost Function (Simple Tracking) ---
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        
        ny_total = ny + nu 
        ocp.dims.ny = ny_total
        ocp.dims.ny_e = ny

        # 
        Vx = np.zeros((ny_total, nx_aug))
        Vx[:ny, :nz] = self.C   # Track output
        ocp.cost.Vx = Vx

        # Terminal cost only on output error
        Vx_e = np.zeros((ny, nx_aug))  
        Vx_e[:ny, :nz] = self.C       
        ocp.cost.Vx_e = Vx_e

        # Vu penalizes change in control input (delta u)
        Vu = np.zeros((ny_total, nu))
        Vu[ny:, :] = np.eye(nu) # Penalize control effort
        ocp.cost.Vu = Vu

        # Weight Matrices
        W = scipy.linalg.block_diag(self.Q, self.R)
        ocp.cost.W = W
        ocp.cost.W_e = self.Q

        # Placeholders for reference
        ocp.cost.yref = np.zeros(ny_total)
        ocp.cost.yref_e = np.zeros(ny)

        # === Constraints (THE SIMPLE PART) ===
        # --- Initial Condition ---
        ocp.constraints.idxbx_0 = np.arange(nx_aug)
        ocp.constraints.lbx_0 = np.zeros(nx_aug) 
        ocp.constraints.ubx_0 = np.zeros(nx_aug) 

        # --- Augmented state constraints ---
        nx_aug = nz + nu
        lbx = np.zeros(nx_aug)
        ubx = np.zeros(nx_aug)

        # Case 1: bound augmented state using lifted velocity limits
        # x_max_phys = self.max_vel.reshape(1, -1)
        # x_max_sc = self.scaler_x.transform(x_max_phys).flatten()

        # z_max_sc = self.lift_func(x_max_sc) 
        # z_min_sc = self.lift_func(-x_max_sc)

        # lbx[:nz] = z_min_sc
        # ubx[:nz] = z_max_sc

        # u_limit_sc = self.scaler_u.transform(np.array([[200.0]*nu]))[0]
        # lbx[nz:] = -u_limit_sc
        # ubx[nz:] = u_limit_sc

        # ocp.constraints.idxbx = np.arange(nx_aug)
        # ocp.constraints.lbx = lbx
        # ocp.constraints.ubx = ubx

        # Case 2: bound output (y = Cz) 
        # [Cz; u_prev] from [z; u_prev]
        # Matrix size: (ny+nu) x (nz+nu)
        D_mat = np.block([
            [self.C, np.zeros((ny, nu))],
            [np.zeros((nu, nz)), np.eye(nu)]
        ])

        v_max_sc = self.scaler_x.transform(self.max_vel.reshape(1, -1))[0]
        tau_max_sc = self.scaler_u.transform(self.max_tau.reshape(1, -1))[0]

        y_max = np.concatenate([v_max_sc, tau_max_sc])
        y_min = -y_max

        # General Linear Constraints (lg <= C*x + D*u <= ug)
        ocp.constraints.C = D_mat 
        ocp.constraints.D = np.zeros((D_mat.shape[0], nu))
        ocp.constraints.lg = y_min
        ocp.constraints.ug = y_max

        # === input of augmented system (delta tau) ===
        delta_tau_max_sc = self.scaler_u.transform(self.max_delta_tau.reshape(1, -1))[0]  # Max change in control input
        ocp.constraints.lbu = np.array(-delta_tau_max_sc)
        ocp.constraints.ubu = np.array(delta_tau_max_sc)
        ocp.constraints.idxbu = np.arange(nu)

        # === Solver Options ===
        ocp.solver_options.tf = self.N * self.dt
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.nlp_solver_max_iter = 1
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' 
        ocp.solver_options.qp_solver_cond_N = 5 
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.print_level = 0
        ocp.solver_options.qp_solver_iter_max = 50
        ocp.solver_options.tol = 1e-4 

        self.solver = AcadosOcpSolver(ocp, json_file=json_file)

    def update_weights(self, Q_diag=None, R_diag=None):
        """Update cost matrices online without rebuilding the solver."""
        if Q_diag is not None:
            self.Q = np.diag(Q_diag)
        if R_diag is not None:
            self.R = np.diag(R_diag)
        
        # Rebuild W matrix (ny_total x ny_total)
        new_W = scipy.linalg.block_diag(self.Q, self.R)
        
        # update running cost weight matrix (0 to N-1)
        for i in range(self.N):
            self.solver.cost_set(i, "W", new_W)
        
        # update Terminal Weight (only Q)
        self.solver.cost_set(self.N, "W", self.Q)
        
        if self.logger:
            self.logger.info("MPC Weights updated online.")

    def compute_control(self, x, y_ref):
        if not self.is_ready: return np.zeros(self.u_prev_scaled.shape)

        # 1. Prepare State
        x_scaled = self.scaler_x.transform(x.reshape(1, -1)).flatten()
        z_scaled = self.lift_func(x_scaled)
        
        # State
        xi_t = np.concatenate([z_scaled, self.u_prev_scaled])

        # 2. Set Initial Condition 
        self.solver.set(0, "lbx", xi_t)
        self.solver.set(0, "ubx", xi_t)

        # 3. Set Reference
        y_ref_scaled = self.scaler_y.transform(y_ref.reshape(1, -1)).flatten()
        yref_vec = np.concatenate([y_ref_scaled, np.zeros(self.u_prev_scaled.size)]) # Ref for State + 0 penalty for Delta u
        
        for i in range(self.N):
            self.solver.set(i, "yref", yref_vec)
        self.solver.set(self.N, "yref", y_ref_scaled) # Terminal ref (no input)

        # 4. Solve
        status = self.solver.solve()
        delta_u_scaled = self.solver.get(0, "u")

        # 5. Integrate & Return
        self.u_prev_scaled += delta_u_scaled
        return self.scaler_u.inverse_transform(self.u_prev_scaled.reshape(1, -1)).flatten()
  
def main(args=None):
    rclpy.init(args=args)
    node = CascadeKoopmanControl()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try: executor.spin()
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()