#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import WrenchStamped
from xplorer_mini_common_interfaces.srv import SetModel
from xplorer_mini_common_interfaces.msg import AuvStatus

import mlflow
import numpy as np
import scipy.sparse as sparse
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat

class CascadeKoopmanControl(Node):
    def __init__(self):
        super().__init__('gnc/control/cascade_kmpc')
        self.get_logger().info('Cascade KMPC Node has been started.')

        # --- 1. System Config ---
        self.dt = 0.1
        self.mlflow_uri = "https://mlflow.amarr.tan" 
        mlflow.set_tracking_uri(self.mlflow_uri)

        # --- 2. State Variables ---
        self.eta_error = np.zeros(6)    # Pose Error [x, y, z, r, p, y]
        self.vel = np.zeros(6)          # Current Velocity [u, v, w, p, q, r]
        
        # --- 3. Parameter Declaration ---
        # PID Gains (Outer Loop)
        self.declare_parameters(namespace='', parameters=[
            ('Kp_pose', [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            ('Ki_pose', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ('Kd_pose', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ])
        
        # MPC Weights (Inner Loop) - Using Diagonals for easy tuning
        self.declare_parameters(namespace='', parameters=[
            ('model_name', 'xplorer_mini_velocity_koopman'),
            ('model_stage', 'Production'),
            ('Q_diag', [20.0, 20.0, 20.0, 10.0, 10.0, 10.0]), # Velocity Tracking Cost
            ('R_diag', [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),      # Control Effort Cost
        ])

        self.add_on_set_parameters_callback(self.parameters_callback)

        # --- 4. Controllers Init ---
        # 4.1 Outer Loop: Position PID
        kp = np.array(self.get_parameter('Kp_pose').value)
        ki = np.array(self.get_parameter('Ki_pose').value)
        kd = np.array(self.get_parameter('Kd_pose').value)
        self.position_controller = PositionController(kp, ki, kd, logger=self.get_logger())

        # 4.2 Inner Loop: Velocity MPC
        q_diag = np.array(self.get_parameter('Q_diag').value)
        r_diag = np.array(self.get_parameter('R_diag').value)
        self.velocity_controller = VelocityController(self.dt, q_diag, r_diag, logger=self.get_logger())
        
        # Load Model initially
        self.load_model_and_setup_mpc()

        # --- 5. ROS Interfaces ---
        self.create_subscription(AuvStatus, 'gnc/control_sync', self.odometry_callback, 10)
        self.create_service(SetModel, 'gnc/koopman/set_model', self.set_model_callback)
        self.wrench_pub = self.create_publisher(WrenchStamped, 'gnc/cmd_wrench/wrench_desired', 10)
        self.create_timer(self.dt, self.timer_callback)
    
    def parameters_callback(self, params):
        update_mpc_weights = False
        new_Q = self.velocity_controller.Q_diag
        new_R = self.velocity_controller.R_diag

        for param in params:
            # PID Update
            if param.name == 'Kp_pose':
                self.position_controller.kp = np.array(param.value)
            elif param.name == 'Ki_pose':
                self.position_controller.ki = np.array(param.value)
            elif param.name == 'Kd_pose':
                self.position_controller.kd = np.array(param.value)
            
            # MPC Weight Update
            elif param.name == 'Q_diag':
                new_Q = np.array(param.value)
                update_mpc_weights = True
            elif param.name == 'R_diag':
                new_R = np.array(param.value)
                update_mpc_weights = True

        if update_mpc_weights and self.velocity_controller.is_ready:
            self.velocity_controller.update_weights(new_Q, new_R)
            self.get_logger().info("MPC Weights Updated via Parameter Server")

        return SetParametersResult(successful=True)
    
    def odometry_callback(self, msg):
        self.eta_error = np.array(msg.eta_e) 
        self.vel = np.array(msg.nu)
    
    def timer_callback(self):
        # Safety Check: If MPC is not ready, do nothing
        if not self.velocity_controller.is_ready:
            return

        # --- CASCADE CONTROL LOOP ---

        # 1. Outer Loop (Position PID)
        # Input: Pose Error -> Output: Desired Velocity (v_ref)
        v_ref = self.position_controller.compute_control(self.eta_error, self.dt)
        
        # 2. Inner Loop (Velocity MPC)
        # Input: Current Vel & Desired Vel -> Output: Thrust (u)
        thrust_cmd = self.velocity_controller.compute_control(self.vel, v_ref)

        # 3. Publish Command
        self.publish_wrench(thrust_cmd)

    def load_model_and_setup_mpc(self):
        name = self.get_parameter('model_name').value
        stage = self.get_parameter('model_stage').value
        try:
            self.get_logger().info(f"Loading MLflow model: {name}/{stage}")
            model_uri = f"models:/{name}/{stage}"
            
            # Load Wrapper and Unwrap
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            wrapper = loaded_model.unwrap_python_model() 
            
            # Setup MPC
            self.velocity_controller.setup_model(wrapper)
            self.get_logger().info("MPC Controller Initialized successfully.")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            return False

    def set_model_callback(self, request, response):
        self.set_parameters([
            rclpy.parameter.Parameter('model_name', rclpy.Parameter.Type.STRING, request.model_type),
            rclpy.parameter.Parameter('model_stage', rclpy.Parameter.Type.STRING, str(request.model_version))
        ])
        success = self.load_model_and_setup_mpc()
        response.success = success
        response.message = "Model updated" if success else "Update failed"
        return response

    def publish_wrench(self, u):
        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = float(u[0]), float(u[1]), float(u[2])
        msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = float(u[3]), float(u[4]), float(u[5])
        self.wrench_pub.publish(msg)


class PositionController:
    """ Simple PID for Outer Loop with Safety Limits """
    def __init__(self, kp, ki, kd, logger=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = np.zeros(6)
        self.prev_error = np.zeros(6)
        self.logger = logger
        # Max velocity limit (m/s, rad/s) - Tune this!
        self.max_vel = np.array([1.5, 0.5, 0.5, 0.5, 0.5, 0.5]) 

    def compute_control(self, error, dt):
        self.integral += error * dt
        # Anti-windup
        self.integral = np.clip(self.integral, -5.0, 5.0) 
        
        derivative = (error - self.prev_error) / dt
        
        # PID Calculation
        v_cmd = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        # Saturation
        v_cmd = np.clip(v_cmd, -self.max_vel, self.max_vel)
        
        self.prev_error = error
        return v_cmd


class VelocityController:
    """ Koopman MPC Wrapper using Acados """
    def __init__(self, dt, Q_diag, R_diag, N_horizon=20, logger=None):
        self.dt = dt
        self.N = N_horizon
        self.Q_diag = Q_diag
        self.R_diag = R_diag
        self.logger = logger
        self.solver = None
        self.is_ready = False
        self.wrapper = None
        self.lift_func = None

    def setup_model(self, wrapper):
        self.wrapper = wrapper
        
        # Extract Matrices & Scalers from MLflow Wrapper
        self.A = wrapper.A
        self.B = wrapper.B
        self.C = wrapper.C
        self.scaler_x = wrapper.scaler_x
        self.scaler_u = wrapper.scaler_u
        self.scaler_y = wrapper.scaler_y 
        
        # Determine Lifting Logic
        if hasattr(wrapper.model, 'encode'): # DeepKoopman (PyTorch)
            # Note: Ensure torch is imported if using DeepKoopman
            self.lift_func = lambda x: wrapper.model.encode(x).detach().cpu().numpy()
        elif hasattr(wrapper.model, 'lift'): # EDMDc
            self.lift_func = wrapper.model.lift
        else: # DMDc
            self.lift_func = lambda x: x

        # Re-build Acados Solver
        self._init_acados_solver()
        self.is_ready = True

    def _init_acados_solver(self):
        # Dimensions
        nz_orig, nu = self.B.shape
        ny = self.C.shape[0]
        
        # Augmented State: x_aug = [z; u_prev] -> dimension: nz_orig + nu
        nx_aug = nz_orig + nu
        # Control Input is now Delta U: u_ctrl = delta_u -> dimension: nu
        
        ocp = AcadosOcp()
        ocp.model.name = 'velocity_kmpc_delta_u'

        # --- Variables ---
        z_orig = SX.sym('z_orig', nz_orig)
        u_prev = SX.sym('u_prev', nu)
        x_aug = vertcat(z_orig, u_prev)
        
        delta_u = SX.sym('delta_u', nu)
        
        # --- Augmented Dynamics ---
        # z_next = A*z + B*(u_prev + delta_u)
        # u_next = u_prev + delta_u
        z_next = self.A @ z_orig + self.B @ (u_prev + delta_u)
        u_curr = u_prev + delta_u
        
        ocp.model.f_expl_expr = vertcat(z_next, u_curr)
        ocp.model.x = x_aug
        ocp.model.u = delta_u

        # --- Cost (Linear Least Squares) ---
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        
        # Mapping: y = [C*z; u_curr; delta_u]
        # ny_total = ny (output) + nu (input value) + nu (input rate)
        ny_total = ny + nu + nu
        
        # Vx matrix: maps [z; u_prev] to y
        ocp.cost.Vx = np.zeros((ny_total, nx_aug))
        ocp.cost.Vx[:ny, :nz_orig] = self.C           # Tracking error (C*z)
        ocp.cost.Vx[ny:ny+nu, nz_orig:] = np.eye(nu) # Input penalty (u_prev) - simplified
        
        # Vu matrix: maps delta_u to y
        ocp.cost.Vu = np.zeros((ny_total, nu))
        ocp.cost.Vu[ny:ny+nu, :] = np.eye(nu)        # u_curr = u_prev + delta_u
        ocp.cost.Vu[ny+nu:, :] = np.eye(nu)          # delta_u penalty
        
        # Weight Matrix (Q, R, S)
        S_diag = self.R_diag * 10 # Example: prioritize smoothness
        W_matrix = sparse.diags(np.concatenate([self.Q_diag, self.R_diag, S_diag])).tocsc()
        W_e_matrix = sparse.diags(self.Q_diag).tocsc()

        ocp.cost.W = W_matrix
        ocp.cost.W_e = W_e_matrix

        # --- Constraints ---
        delta_u_max = 0.1 # Engineering limit for rate of change
        ocp.constraints.lbu = np.array([-delta_u_max] * nu)
        ocp.constraints.ubu = np.array([delta_u_max] * nu)
        ocp.constraints.idxbu = np.arange(nu)
        
        # Initial state augmented system
        ocp.constraints.x0 = np.zeros(nx_aug)

        # --- Solver Options ---
        ocp.dims.N = self.N
        ocp.solver_config.tf = self.N * self.dt
        ocp.solver_config.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_config.nlp_solver_type = 'SQP_RTI'
        ocp.solver_config.integrator_type = 'DISCRETE'
        
        self.solver = AcadosOcpSolver(ocp, json_file='acados_kmpc_augmented.json')

    def update_weights(self, new_Q, new_R):
        """ Update Cost Weights at Runtime """
        self.Q_diag = new_Q
        self.R_diag = new_R
        
        if self.solver is None: return

        W_new = np.diag(np.concatenate([new_Q, new_R]))
        W_e_new = np.diag(new_Q)

        for i in range(self.N):
            self.solver.cost_set(i, "W", W_new)
        self.solver.cost_set(self.N, "W", W_e_new)

    def compute_control(self, v_curr_raw, v_ref_raw):
        if not self.is_ready: return np.zeros(6)

        try:
            # 1. Scale & Lift Current State
            v_curr_scaled = self.scaler_x.transform(v_curr_raw.reshape(1, -1))
            z_curr = self.lift_func(v_curr_scaled).flatten()
            
            # 2. Scale Reference
            v_ref_scaled = self.scaler_y.transform(v_ref_raw.reshape(1, -1)).flatten()

            # 3. Setup Solver Inputs
            self.solver.set(0, "lbx", z_curr)
            self.solver.set(0, "ubx", z_curr)
            
            # Set Reference (Tracking)
            u_ref = np.zeros(self.B.shape[1])
            y_ref = np.concatenate([v_ref_scaled, u_ref])
            
            for i in range(self.N):
                self.solver.set(i, "yref", y_ref)
            self.solver.set(self.N, "yref", v_ref_scaled)

            # 4. Solve
            status = self.solver.solve()
            
            if status != 0:
                return np.zeros(6) # Failsafe

            # 5. Get Output & Inverse Scale
            u_scaled = self.solver.get(0, "u").reshape(1, -1)
            u_physical = self.scaler_u.inverse_transform(u_scaled).flatten()
            return u_physical

        except Exception as e:
            if self.logger:
                self.logger.error(f"MPC Solve Failed: {e}")
            return np.zeros(6)

def main(args=None):
    rclpy.init(args=args)
    node = CascadeKoopmanControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()