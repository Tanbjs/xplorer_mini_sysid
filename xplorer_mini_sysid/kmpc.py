#!/usr/bin/env python3
import logging
import warnings
import os
import shutil
import time

# --- 0. Environment Setup & Safety ---
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
import scipy.linalg
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat, DM, mtimes

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
        self.eta_error = np.zeros(6)
        self.vel = np.zeros(6)
        
        # --- 3. Parameter Declaration ---
        self.declare_parameters(namespace='', parameters=[
            ('Kp_pose', [1.0] * 6),
            ('Ki_pose', [0.0] * 6),
            ('Kd_pose', [0.0] * 6),
            ('integral_limit', [5.0] * 6),
            ('max_vel', [1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
            ('model_name', 'dmdc'),
            ('Q_diag', [20.0, 20.0, 20.0, 10.0, 10.0, 10.0]),
            ('R_diag', [0.1] * 6),
            ('S_diag', [1.0] * 6)
        ])

        self.add_on_set_parameters_callback(self.parameters_callback)

        # --- 4. Controllers Init ---
        self.position_controller = PositionController(
            kp=np.array(self.get_parameter('Kp_pose').value),
            ki=np.array(self.get_parameter('Ki_pose').value),
            kd=np.array(self.get_parameter('Kd_pose').value),
            int_limit=np.array(self.get_parameter('integral_limit').value),
            max_vel=np.array(self.get_parameter('max_vel').value),
            logger=self.get_logger()
        )

        q_diag = np.array(self.get_parameter('Q_diag').value)
        r_diag = np.array(self.get_parameter('R_diag').value)
        s_diag = np.array(self.get_parameter('S_diag').value)

        self.velocity_controller = VelocityController(
            dt=self.dt, 
            Q_diag=q_diag, 
            R_diag=r_diag, 
            S_diag=s_diag,
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
            if param.name == 'Kp_pose':
                self.position_controller.kp = np.array(param.value)
            elif param.name == 'Ki_pose':
                self.position_controller.ki = np.array(param.value)
            elif param.name == 'Kd_pose':
                self.position_controller.kd = np.array(param.value)
            elif param.name == 'integral_limit':
                self.position_controller.int_limit = np.array(param.value)
            elif param.name == 'max_vel':
                self.position_controller.max_vel = np.array(param.value)
            elif param.name == 'Q_diag':
                self.velocity_controller.Q_diag = np.array(param.value)
                self.velocity_controller.update_weights() # Trigger update
            elif param.name == 'R_diag':
                self.velocity_controller.R_diag = np.array(param.value)
                self.velocity_controller.update_weights() # Trigger update
            elif param.name == 'S_diag':
                self.velocity_controller.S_diag = np.array(param.value)
                self.velocity_controller.update_weights() # Trigger update

        return SetParametersResult(successful=True)
    
    def odometry_callback(self, msg):
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
        v_ref = self.position_controller.compute_control(self.eta_error, self.dt)
        
        # 2. Inner Loop
        # IMPORTANT: Pass current velocity and reference velocity
        thrust_cmd = self.velocity_controller.compute_control(self.vel, v_ref)

        # 3. Publish
        self.publish_wrench(thrust_cmd)
        self.publish_twist(self.twist_pub, v_ref) # Debug: Publish desired twist

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


class PositionController:
    def __init__(self, kp, ki, kd, int_limit, max_vel, logger=None):
        self.kp = kp; self.ki = ki; self.kd = kd; self.int_limit = int_limit
        self.integral = np.zeros(6)
        self.prev_error = np.zeros(6)
        self.max_vel = np.array(max_vel) 

    def compute_control(self, error, dt):
        if dt <= 0: return np.zeros(6)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.int_limit, self.int_limit) 
        derivative = (error - self.prev_error) / dt
        v_cmd = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        v_cmd = np.clip(v_cmd, -self.max_vel, self.max_vel)
        self.prev_error = error
        return v_cmd


class VelocityController:
    def __init__(self, dt, Q_diag, R_diag, S_diag, node_name, N_horizon=20, logger=None):
        self.dt = dt
        self.N = N_horizon
        self.Q_diag = Q_diag
        self.R_diag = R_diag
        self.S_diag = S_diag
        self.node_name = node_name
        self.logger = logger
        self.solver = None
        self.is_ready = False
        self.u_prev_scaled = None # Keep track of scaled control action

    def setup_model(self, wrapper):
        self.wrapper = wrapper
        self.A = wrapper.A; self.B = wrapper.B; self.C = wrapper.C
        # SCALERS MUST BE AVAILABLE
        self.scaler_x = getattr(wrapper, 'scaler_x', None) # Not used in KMPC usually if lifted
        self.scaler_u = wrapper.scaler_u
        self.scaler_y = wrapper.scaler_y 
        
        new_nu = self.B.shape[1]
        
        # Reset previous control when model changes to avoid jump
        self.u_prev_scaled = np.zeros(new_nu)
            
        if self.logger: self.logger.info(f"Model Set: nu={new_nu}")

        if hasattr(wrapper.model, 'encode'):
            self.lift_func = lambda x: wrapper.model.encode(x).detach().cpu().numpy()
        elif hasattr(wrapper.model, 'lift'):
            self.lift_func = wrapper.model.lift
        else:
            self.lift_func = lambda x: x

        self._init_acados_solver()
        self.is_ready = True

    def _init_acados_solver(self):
        # 1. Clean previous build
        generated_path = f'c_generated_code_{self.node_name}'
        if os.path.exists(generated_path):
            shutil.rmtree(generated_path)

        # 2. Data Prepare
        try:
            raw_A = np.array(self.A, dtype=np.float64).flatten()
            raw_B = np.array(self.B, dtype=np.float64).flatten()
            raw_C = np.array(self.C, dtype=np.float64).flatten()

            nz_orig = int(np.sqrt(len(raw_A))) 
            nu = int(len(raw_B) / nz_orig)
            ny = int(len(raw_C) / nz_orig)
            nx_aug = nz_orig + nu

            np_A = np.reshape(raw_A, (nz_orig, nz_orig), order='C')
            np_B = np.reshape(raw_B, (nz_orig, nu), order='C')
            np_C = np.reshape(raw_C, (ny, nz_orig), order='C')

            if self.logger:
                self.logger.info(f"Acados Shapes -> nz:{nz_orig}, nu:{nu}, ny:{ny}")
                
            self.dims = {'nz': nz_orig, 'nu': nu, 'ny': ny, 'nx_aug': nx_aug} # Save dims for later use

        except Exception as e:
            if self.logger: self.logger.error(f"Prep Error: {e}")
            return

        # 3. Setup OCP
        ocp = AcadosOcp()
        ocp.code_export_directory = generated_path
        ocp.model.name = f'kmpc_{self.node_name}'

        sym_x = SX.sym('x', nx_aug, 1)
        sym_u = SX.sym('u', nu, 1) # delta_u

        ocp.model.x = sym_x
        ocp.model.u = sym_u

        # --- Augmented Dynamics ---
        A_aug = np.block([[np_A, np_B], [np.zeros((nu, nz_orig)), np.eye(nu)]])
        B_aug = np.block([[np_B], [np.eye(nu)]])
        
        ocp.model.disc_dyn_expr = mtimes(SX(DM(A_aug)), sym_x) + mtimes(SX(DM(B_aug)), sym_u)

        # --- Dimensions ---
        ny_total = ny + nu + nu
        ocp.dims.nx = nx_aug
        ocp.dims.nu = nu
        ocp.dims.ny = ny_total
        ocp.dims.ny_e = ny 

        # --- Cost ---
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        
        def to_c(arr): return np.ascontiguousarray(arr, dtype=np.float64)

        Vx = np.zeros((ny_total, nx_aug))
        Vx[:ny, :nz_orig] = np_C              
        Vx[ny+nu:, nz_orig:] = np.eye(nu)  
        ocp.cost.Vx = to_c(Vx)

        Vu = np.zeros((ny_total, nu))
        Vu[ny:ny+nu, :] = np.eye(nu)         
        Vu[ny+nu:, :] = np.eye(nu)           
        ocp.cost.Vu = to_c(Vu)
        
        Vx_e = np.zeros((ny, nx_aug))
        Vx_e[:ny, :nz_orig] = np_C
        ocp.cost.Vx_e = to_c(Vx_e)

        # Weights
        W_block = [np.diag(self.Q_diag), np.diag(self.R_diag), np.diag(self.S_diag)]
        W_matrix = scipy.linalg.block_diag(*W_block)
        W_e_matrix = np.diag(self.Q_diag)

        ocp.cost.W = to_c(W_matrix)
        ocp.cost.W_e = to_c(W_e_matrix)
        
        ocp.cost.yref = to_c(np.zeros(ny_total))
        ocp.cost.yref_e = to_c(np.zeros(ny))

        # --- Constraints ---
        # NOTE: Check if scaler_u is available to scale delta_u limits if needed. 
        # For now, assuming delta_u_max is in scaled domain or robust enough.
        delta_u_max = 0.5 
        ocp.constraints.lbu = np.array([-delta_u_max] * nu)
        ocp.constraints.ubu = np.array([delta_u_max] * nu)
        ocp.constraints.idxbu = np.arange(nu)
        ocp.constraints.x0 = np.zeros(nx_aug)

        # --- Options ---
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.N * self.dt
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.print_level = 0
        
        json_file = f'acados_ocp_{self.node_name}.json'
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)

    def update_weights(self):
        """Updates the cost weights in the active solver."""
        if self.solver is None: return

        # Reconstruct W matrix with current Q, R, S
        W_block = [np.diag(self.Q_diag), np.diag(self.R_diag), np.diag(self.S_diag)]
        W_new = scipy.linalg.block_diag(*W_block)
        W_e_new = np.diag(self.Q_diag)
        
        # Ensure C-contiguous
        W_new = np.ascontiguousarray(W_new, dtype=np.float64)
        W_e_new = np.ascontiguousarray(W_e_new, dtype=np.float64)

        try:
            for i in range(self.N): 
                self.solver.cost_set(i, "W", W_new)
            self.solver.cost_set(self.N, "W", W_e_new)
            if self.logger: self.logger.info("MPC Weights updated in solver.")
        except Exception as e:
            if self.logger: self.logger.error(f"Failed to update weights: {e}")

    def compute_control(self, v_curr_raw, v_ref_raw):
        if not self.is_ready: return np.zeros(6)
        
        try:
            # 1. Scale Inputs
            # transform() needs 2D array (1, n_features)
            v_curr_scaled = self.scaler_y.transform(v_curr_raw.reshape(1, -1))
            
            # 2. Lift State (Koopman)
            # Input: Scaled measurement -> Output: Lifted state z
            z_curr = self.lift_func(v_curr_scaled).flatten() 
            
            # 3. Prepare Initial State x0_aug = [z_k; u_{k-1}]
            # Note: u_prev_scaled is already scaled
            x0_aug = np.concatenate([z_curr, self.u_prev_scaled])

            # 4. Prepare Reference
            v_ref_scaled = self.scaler_y.transform(v_ref_raw.reshape(1, -1)).flatten()
            
            # 5. Set Initial Condition
            self.solver.set(0, "lbx", x0_aug)
            self.solver.set(0, "ubx", x0_aug)
            
            # 6. Set References
            # Cost vector y = [y_out; delta_u; u_act]
            # We want y_out -> v_ref, delta_u -> 0, u_act -> 0 (or maintain)
            nu = self.dims['nu']
            y_ref = np.concatenate([v_ref_scaled, np.zeros(nu), np.zeros(nu)])
            y_ref_e = v_ref_scaled # Terminal reference

            for i in range(self.N): 
                self.solver.set(i, "yref", y_ref)
            self.solver.set(self.N, "yref", y_ref_e)

            # 7. Solve
            status = self.solver.solve()
            if status != 0: 
                if self.logger: self.logger.warn(f"Acados returned status {status}")
                # Fallback: return previous real control or zero?
                # Using zero might be safer than stuck throttle in underwater
                return np.zeros(6) 

            # 8. Get Result (Delta U)
            delta_u_scaled = self.solver.get(0, "u")
            
            # 9. Integrate & Unscale
            u_next_scaled = self.u_prev_scaled + delta_u_scaled
            
            # Update state for next loop
            self.u_prev_scaled = u_next_scaled 
            
            # Convert to physical units (Newtons/Torque)
            u_real = self.scaler_u.inverse_transform(u_next_scaled.reshape(1, -1)).flatten()
            
            return u_real

        except Exception as e:
            if self.logger: self.logger.error(f"MPC Compute Fail: {e}")
            return np.zeros(6)

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