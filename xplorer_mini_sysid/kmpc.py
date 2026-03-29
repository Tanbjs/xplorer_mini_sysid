#!/usr/bin/env python3
import logging
import warnings
import os
import time
from xmlrpc import client
import matplotlib

matplotlib.use('Agg')
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

import rclpy
from rclpy.publisher import Publisher
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult
from message_filters import Subscriber, ApproximateTimeSynchronizer

from geometry_msgs.msg import WrenchStamped, TwistStamped
from nav_msgs.msg import Path, Odometry
from std_srvs.srv import Trigger 
from gazebo_msgs.srv import SetEntityState
from xplorer_mini_common_interfaces.srv import SetModel
from xplorer_mini_common_interfaces.msg import AuvStatus

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np

from xplorer_mini_python_utils.kinematics import eulerang, odom_to_state_vect, pose_msg_to_vect, ssa
from xplorer_mini_sysid.lib.core.params import Weights, Bounds
from xplorer_mini_sysid.lib.controller import mpc, lqr
from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper


class CascadeKoopmanControl(Node):
    def __init__(self):
        super().__init__('gnc_control_cascade_kmpc')
        self.get_logger().info('Cascade KMPC Node has been started.')

        # --- System Config ---
        self.is_vel_ready = False
        self.dt = 0.1
        self.mlflow_uri = "https://mlflow.amarr.tan" 
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.mlflow_client = MlflowClient(tracking_uri=self.mlflow_uri)

        # --- Parameter Declaration ---
        self.declare_parameters(namespace='', parameters=[
            ('model_name', 'dmdc'),
            ('model_version', 1),
            ('use_preview', False),
            ('vel_controller', 'mpc_standard_state'), 
            ('vel_mode', 'state'),
            ('rigid_body_mass', np.zeros(6).tolist()),
            ('Kp_pos', [1.0] * 6),
            ('Ki_pos', [0.0] * 6),
            ('Kd_pos', [0.0] * 6),
            ('integral_limit', [5.0] * 6),
            ('max_vel', [1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
            ('N_horizon', 10),
            ('Q_diag', [20.0, 20.0, 20.0, 10.0, 10.0, 10.0]),
            ('R_rate_diag', [10.0] * 6), 
            ('R_abs_diag', [0.1] * 6),
            ('max_tau', [200.0] * 6),
            ('max_delta_tau', [50.0] * 6)
        ])

        self.vel_controller = self.get_parameter('vel_controller').value
        self.vel_mode = self.get_parameter('vel_mode').value
        self.N_horizon = self.get_parameter('N_horizon').value
        self.use_preview = self.get_parameter('use_preview').value
        self.add_on_set_parameters_callback(self.parameters_callback)

        # --- ROS Event ---
        # self.create_subscription(AuvStatus, 'gnc/control_sync', self.control_callback, qos_profile_sensor_data)
        self.odom_sub = Subscriber(self, Odometry, 'gnc/odom_filtered', qos_profile=qos_profile_sensor_data)
        self.path_sub = Subscriber(self, Path, 'gnc/ref_trajectory/window')
        self.ts = ApproximateTimeSynchronizer([self.odom_sub, self.path_sub], queue_size=100, slop=0.15)
        self.ts.registerCallback(self.sync_callback)

        self.set_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.create_service(Trigger, 'gnc/koopman/sim_vel', self.sim_vel_callback)

        self.wrench_pub = self.create_publisher(WrenchStamped, 'gnc/cmd_wrench/wrench_desired', 10)
        self.twist_pub = self.create_publisher(TwistStamped, 'gnc/sysid/twist_desired', 10)
        self.err_twist_pub = self.create_publisher(TwistStamped, 'gnc/sysid/twist_error', 10)

        # --- State Variables ---
        # State vectors at current time step
        self.eta = np.zeros(6)
        self.nu = np.zeros(6)
        self.eta_error = np.zeros(6)

        # Reference windows for cascade control (outer loop provides position reference, inner loop tracks velocity reference)
        self.nu_cmd_b = np.zeros(6)  
        self.eta_cmd_window = np.zeros((self.N_horizon, 6))
        self.nu_hat_cmd_b_window = np.zeros((self.N_horizon, 6))  

        # for sim trajectory
        self.control_mode = 1 # 0: IDLE, 1: NORMAL CONTROL, 2: SIM TRAJECTORY
        self.sim_index = 0              
        self.t_final = 60.0

        # Position controller
        self.pose_ct = PIDController(
            kp=np.array(self.get_parameter('Kp_pos').value),
            ki=np.array(self.get_parameter('Ki_pos').value),
            kd=np.array(self.get_parameter('Kd_pos').value),
            int_limit=np.array(self.get_parameter('integral_limit').value),
            max_vel=np.array(self.get_parameter('max_vel').value),
        )
        self.pose_ct_virtual = PIDController(
            kp=np.array(self.get_parameter('Kp_pos').value),
            ki=np.array(self.get_parameter('Ki_pos').value),
            kd=np.array(self.get_parameter('Kd_pos').value),
            int_limit=np.array(self.get_parameter('integral_limit').value),
            max_vel=np.array(self.get_parameter('max_vel').value),
        )
        # Velocity controller
        self.wrapper = self.load_model()    # load kopman model wrapper from MLflow
        self.vel_ct = self.init_vel_ct()  # initialize MPC controller with loaded model
    
    def parameters_callback(self, params):

        for param in params:
            if param.name == 'Kp_pos':
                self.pose_ct.kp = np.array(param.value)
                self.get_logger().info(f"Updated Kp_pos: {self.pose_ct.kp}")
            
            elif param.name == 'Ki_pos':
                self.pose_ct.ki = np.array(param.value)
                self.get_logger().info(f"Updated Ki_pos: {self.pose_ct.ki}")
            
            elif param.name == 'Kd_pos':
                self.pose_ct.kd = np.array(param.value)
                self.get_logger().info(f"Updated Kd_pos: {self.pose_ct.kd}")
            
            elif param.name == 'integral_limit':
                self.pose_ct.int_limit = np.array(param.value)
                self.get_logger().info(f"Updated integral limits: {self.pose_ct.int_limit}")
            
            elif param.name == 'max_vel':
                self.pose_ct.max_vel = np.array(param.value)
                self.vel_ct.set_params(y_min=-self.pose_ct.max_vel, y_max=self.pose_ct.max_vel) # Update constraints online
                self.get_logger().info(f"Updated max velocity limits: {self.pose_ct.max_vel}")
            
            elif param.name == 'Q_diag':
                self.vel_ct.set_params(Q=np.diag(np.array(param.value)))
                self.get_logger().info(f"Updated Q_diag: {np.diag(self.vel_ct.params.weights.Q)}")
            
            elif param.name == 'R_abs_diag':
                self.vel_ct.set_params(R_abs=np.diag(np.array(param.value)))
                self.get_logger().info(f"Updated R_abs_diag: {np.diag(self.vel_ct.params.weights.R_abs)}")
            
            elif param.name == 'R_rate_diag':
                self.vel_ct.set_params(R_rate=np.diag(np.array(param.value)))
                self.get_logger().info(f"Updated R_rate_diag: {np.diag(self.vel_ct.params.weights.R_rate)}")
            
            elif param.name == 'N_horizon':
                self.vel_ct.set_params(N_horizon=param.value)
                self.get_logger().info(f"Updated MPC horizon: {self.vel_ct.params.N_horizon}")
            
            elif param.name == 'max_tau':
                self.vel_ct.set_params(u_max=np.array(param.value))
                self.get_logger().info(f"Updated max control input limits: {self.vel_ct.params.bounds.u_max}")
            
            elif param.name == 'max_delta_tau':
                self.vel_ct.set_params(du_max=np.array(param.value))
                self.get_logger().info(f"Updated max change in control input limits: {self.vel_ct.params.bounds.du_max}")
            
            elif param.name == "rigid_body_mass":
                self.get_logger().info(f"Received new mass parameters: {param.value}")
                self.m_rb = np.array(param.value)
            
            elif param.name == "vel_controller":
                self.get_logger().info(f"Switching velocity controller to: {param.value}")
                self.vel_controller = param.value

            elif param.name == "include_abs_input":
                self.vel_ct._include_absolute_input = param.value
                self.get_logger().info(f"Updated include_abs_input flag: {self.vel_ct._include_absolute_input}")

        return SetParametersResult(successful=True)
    
    def sim_vel_callback(self, request, response):
        t = np.linspace(0, self.t_final, int(self.t_final/self.dt))
        u = 0.5 * np.sin(2 * np.pi * t / self.t_final)
        v = 0.5 * np.cos(2 * np.pi * t / self.t_final) 
        w = -0.05 * np.sin(2 * np.pi * t / self.t_final) - 0.1
        p = np.zeros_like(u)
        q = np.zeros_like(u)
        r = np.zeros_like(u)
            
        if self.set_state_client.wait_for_service(timeout_sec=1.0):
            state_msg = SetEntityState.Request()
            state_msg.state.name = 'xplorer_mini' 
            state_msg.state.reference_frame = 'world'
            state_msg.state.pose.position.x = 0.0
            state_msg.state.pose.position.y = 0.0
            state_msg.state.pose.position.z = -0.2 

            state_msg.state.twist.linear.x = float(u[0])
            state_msg.state.twist.linear.y = float(v[0])
            state_msg.state.twist.linear.z = float(w[0])
            
            self.set_state_client.call_async(state_msg)
            self.get_logger().info("Gazebo state fixed to trajectory start point.")
        else:
            self.get_logger().error("Gazebo service not available, state not fixed.")

        self.v_cmd_b = np.vstack((u, v, w, p, q, r)).T
        self.sim_index = 0  
        self.control_mode = 2
        response.success = True
        response.message = "Simulation trajectory generated."
        return response
        
    def control_callback(self, msg):
        eta = np.array(msg.eta)
        eta_error = np.array(msg.eta_e) 
        vel = np.array(msg.nu)
        t_start = time.time()

        if not self.is_vel_ready:
            return

        # MODE SELECTION & REFERENCE GENERATION ---
        if self.control_mode == 2:  # [MODE 2] SIM TRAJECTORY
            
            know_future = True 
            traj_chunk = self.v_cmd_b[self.sim_index : self.sim_index + self.N_horizon]
            
            if traj_chunk.shape[0] < self.N_horizon:
                pad_size = self.N_horizon - traj_chunk.shape[0]
                last_val = traj_chunk[-1] if traj_chunk.size > 0 else np.zeros(6)
                v_cmd_b_full = np.vstack([traj_chunk, np.tile(last_val, (pad_size, 1))])
            else:
                v_cmd_b_full = traj_chunk

            if know_future:
                v_cmd_b = v_cmd_b_full
            else:
                v_cmd_b = np.tile(v_cmd_b_full[0], (self.N_horizon, 1))

            self.sim_index += 1
            if self.sim_index >= len(self.v_cmd_b):
                self.get_logger().info("Sim finished. Moving to IDLE.")
                self.control_mode = 0 
                self.sim_index = 0

        elif self.control_mode == 1:  # [MODE 1] NORMAL (DOUBLE LOOP)
            v_cmd_n = self.pose_ct.compute_control(-eta_error, self.dt)
            J, _, _ = eulerang(eta[3], eta[4], eta[5])
            v_cmd_b_single = np.linalg.inv(J) @ v_cmd_n
            v_cmd_b = np.tile(v_cmd_b_single, (self.N_horizon, 1))
            # v_cmd_b = np.tile(v_cmd_n, (self.N_horizon, 1))

        else:  # [MODE 0] IDLE / DEFAULT
            self.publish_wrench(np.zeros(6))
            self.publish_twist(self.twist_pub, np.zeros(6))
            return 
        
        # --- 2. INNER LOOP (MPC) ---
        if self.vel_controller == 'mpc_aug_error_output':
            tau_cmd = self.vel_ct.compute_control(vel, vel, v_cmd_b)
        else:
            tau_cmd = self.vel_ct.compute_control(vel, v_cmd_b)
            
        tau_cmd = np.clip(np.asarray(tau_cmd).flatten(), -200, 200)

        # --- 3. PUBLISH ---
        current_v_ref = v_cmd_b[0]
        self.publish_wrench(tau_cmd)
        self.publish_twist(self.twist_pub, current_v_ref)
        self.publish_twist(self.err_twist_pub, current_v_ref - vel)  

    def sync_callback(self, odom_msg, path_msg: Path):
        
        # get current state and reference from messages
        self.eta = odom_to_state_vect(odom_msg).flatten()[:6]
        self.nu = odom_to_state_vect(odom_msg).flatten()[6:12]
        for i in range(len(path_msg.poses)):
            self.eta_cmd_window[i, :] = pose_msg_to_vect(path_msg.poses[i].pose).flatten()
        
        # calculate cascade control commands
        J, _, _ = eulerang(self.eta[3], self.eta[4], self.eta[5])
        J_inv = np.linalg.inv(J)
        self.nu_cmd_b = J_inv @ self.pose_ct.compute_control(self.eta_cmd_window[0] - self.eta, self.dt)
        self.nu_hat_cmd_b_window = self.generate_virtual_reference(self.eta_cmd_window, self.eta, self.nu_cmd_b)
        if self.use_preview:
            self.tau_cmd = self.vel_ct.compute_control(self.nu, self.nu_hat_cmd_b_window)
        else:
            self.tau_cmd = self.vel_ct.compute_control(self.nu, self.nu_hat_cmd_b_window[0]) 

        # publish control commands
        self.publish_wrench(self.tau_cmd)
        self.publish_twist(self.twist_pub, self.nu_cmd_b)
        self.publish_twist(self.err_twist_pub, self.nu_cmd_b - self.nu)  

    def generate_virtual_reference(self, eta_cmd_window, eta, nu_cmd_b):
        
        # Preallocate 
        nu_hat_cmd_b_window = np.zeros((self.N_horizon, 6))
        eta_hat = np.copy(eta)
        nu_hat_cmd_b_window[0] = nu_cmd_b

        # Get current outer loop integral and derivative states for continuity in the virtual controller
        self.pose_ct_virtual.integral = np.copy(self.pose_ct.integral)
        self.pose_ct_virtual.prev_error = np.copy(self.pose_ct.prev_error)
        
        for i in range(1, self.N_horizon):
            eta_ref = eta_cmd_window[i]
            eta_error = np.zeros(6)
            eta_error[:3] = eta_ref[:3] - eta_hat[:3]
            for j in range(3):
                eta_error[3+j] = ssa(eta_ref[3+j], eta_hat[3+j])

            J, _, _ = eulerang(eta_hat[3], eta_hat[4], eta_hat[5])
            J_inv = np.linalg.inv(J)
            nu_cmd_n = self.pose_ct_virtual.compute_control(eta_error, self.dt)
            nu_hat_cmd_b_window[i] = J_inv @ nu_cmd_n
            eta_hat = eta_hat + self.dt * nu_cmd_n

        return nu_hat_cmd_b_window

    def load_model(self) -> DMDcWrapper | EDMDcWrapper | DeepModelWrapper:
        name = self.get_parameter('model_name').value
        version = self.get_parameter('model_version').value
        try:
            model = self.mlflow_client.get_model_version(name, version)
            loaded_model = mlflow.pyfunc.load_model(model.source).unwrap_python_model()
            return loaded_model

        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")

    def publish_wrench(self, tau: np.ndarray):
        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = float(tau[0]), float(tau[1]), float(tau[2])
        msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = float(tau[3]), float(tau[4]), float(tau[5])
        self.wrench_pub.publish(msg)

    def publish_twist(self, publisher: Publisher, nu: np.ndarray):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = float(nu[0]), float(nu[1]), float(nu[2])
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = float(nu[3]), float(nu[4]), float(nu[5])
        publisher.publish(msg)
    
    def init_vel_ct(self):

        if self.vel_controller == 'lqr_standard':
            weights = Weights(
                Q=np.diag(self.get_parameter('Q_diag').value),
                R_abs=np.diag(self.get_parameter('R_abs_diag').value),
            )
            lqr_params = lqr.LQRParams(weights=weights)
            vel_ct = lqr.Standard(model_wrapper=self.wrapper, 
                                  lqr_params=lqr_params, 
                                  node_name=self.get_name(), 
                                  use_preview=self.use_preview,
                                  logger=self.get_logger())
        
        elif self.vel_controller == 'mpc_standard':

            weights = Weights(
                Q=np.diag(self.get_parameter('Q_diag').value),
                R_abs=np.diag(self.get_parameter('R_abs_diag').value),
            )
            bounds = Bounds(
                u_min= -np.array(self.get_parameter('max_tau').value),
                u_max= np.array(self.get_parameter('max_tau').value),
                du_min= -np.array(self.get_parameter('max_delta_tau').value),
                du_max= np.array(self.get_parameter('max_delta_tau').value),
                y_min= -np.array(self.get_parameter('max_vel').value),
                y_max= np.array(self.get_parameter('max_vel').value)
            )
            mpc_params = mpc.MPCParams(
                dt=self.dt,
                N_horizon=self.get_parameter('N_horizon').value,
                weights=weights,
                bounds=bounds
            )
            vel_ct = mpc.Standard(mode=self.vel_mode,
                                  model_wrapper=self.wrapper, 
                                  mpc_params=mpc_params, 
                                  node_name=self.get_name(), 
                                  use_preview=self.use_preview,
                                  logger=self.get_logger())

        elif self.vel_controller  == 'mpc_incremental':
            weights = Weights(
                Q=np.diag(self.get_parameter('Q_diag').value),
                R_abs=np.diag(self.get_parameter('R_abs_diag').value),
                R_rate=np.diag(self.get_parameter('R_rate_diag').value)
            )
            bounds = Bounds(
                u_min= -np.array(self.get_parameter('max_tau').value),
                u_max= np.array(self.get_parameter('max_tau').value),
                du_min= -np.array(self.get_parameter('max_delta_tau').value),
                du_max= np.array(self.get_parameter('max_delta_tau').value),
                y_min= -np.array(self.get_parameter('max_vel').value),
                y_max= np.array(self.get_parameter('max_vel').value)
            )
            mpc_params = mpc.MPCParams(
                dt=self.dt,
                N_horizon=self.get_parameter('N_horizon').value,
                weights=weights,
                bounds=bounds
            )
            vel_ct = mpc.Incremental(mode=self.vel_mode, 
                                     model_wrapper=self.wrapper, 
                                     mpc_params=mpc_params, 
                                     node_name=self.get_name(), 
                                     logger=self.get_logger())
        
        elif self.vel_controller == 'mpc_velocity_form':
            weights = Weights(
                Q=np.diag(self.get_parameter('Q_diag').value),
                R_abs=np.diag(self.get_parameter('R_abs_diag').value),
                R_rate=np.diag(self.get_parameter('R_rate_diag').value)
            )
            bounds = Bounds(
                u_min= -np.array(self.get_parameter('max_tau').value),
                u_max= np.array(self.get_parameter('max_tau').value),
                du_min= -np.array(self.get_parameter('max_delta_tau').value),
                du_max= np.array(self.get_parameter('max_delta_tau').value),
                y_min= -np.array(self.get_parameter('max_vel').value),
                y_max= np.array(self.get_parameter('max_vel').value)
            )
            mpc_params = mpc.MPCParams(
                dt=self.dt,
                N_horizon=self.get_parameter('N_horizon').value,
                weights=weights,
                bounds=bounds
            )
            vel_ct = mpc.VelocityForm(model_wrapper=self.wrapper, 
                                      mpc_params=mpc_params, 
                                      node_name=self.get_name(), 
                                      logger=self.get_logger())
        
        elif self.vel_controller == 'mpc_implicit_rigid_tube':
            pass # Placeholder for tube MPC initialization, can be implemented similarly with additional tube parameters

        else:
            raise ValueError(f"Unsupported MPC algorithm: {self.vel_controller}")
        
        self.is_vel_ready = True
        return vel_ct


class PIDController:
    def __init__(self, kp, ki, kd, int_limit, max_vel):
        self.kp = kp; self.ki = ki; self.kd = kd; self.int_limit = int_limit
        self.integral = np.zeros(6)
        self.prev_error = np.zeros(6)
        self.max_vel = np.array(max_vel) 

    def compute_control(self, error, dt):
        if dt <= 0: return np.zeros(6)
        
        derivative = (error - self.prev_error) / dt
        u_p = self.kp * error
        u_d = self.kd * derivative
        u_unsat = u_p + (self.ki * self.integral) + u_d

        # Saturation
        u_cmd = np.clip(u_unsat, -self.max_vel, self.max_vel)
        
        # Anti-windup: Only integrate if we're not saturated in the direction of the error
        is_saturated = (u_unsat > self.max_vel) | (u_unsat < -self.max_vel)
        same_direction = np.sign(error) == np.sign(u_unsat)
        stop_integrating = is_saturated & same_direction
        self.integral += (~stop_integrating) * (error * dt)
        self.integral = np.clip(self.integral, -self.int_limit, self.int_limit)

        self.prev_error = error

        return u_cmd


class PI_FF_Controller:
    def __init__(self, kp, ki):
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.integral = np.zeros(6)

    def compute_control(self, error, v_ff, dt):
        """
        error: Position error (setpoint - current)
        v_ff: Feedforward velocity
        dt: Time step
        """
        self.integral += error * dt
        v_n = v_ff + (self.kp * error) + (self.ki * self.integral)

        return v_n
    
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
