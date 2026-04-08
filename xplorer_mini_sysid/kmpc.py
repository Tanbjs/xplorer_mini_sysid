#!/usr/bin/env python3
import logging
import warnings
import os
import time
import matplotlib
from pytest import param
import yaml
import json

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

from geometry_msgs.msg import WrenchStamped, TwistStamped, PoseStamped
from nav_msgs.msg import Path, Odometry
from std_srvs.srv import Trigger 
from gazebo_msgs.srv import SetEntityState
from xplorer_mini_common_interfaces.srv import SetModel
from xplorer_mini_common_interfaces.msg import AuvStatus

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np

from xplorer_mini_python_utils.kinematics import eulerang, odom_to_state_vect, pose_msg_to_vect, pose_vect_to_msg
from xplorer_mini_sysid.lib.core.params import Weights, Bounds
from xplorer_mini_sysid.lib.utils.kinematic import cal_eta_err_with_ssa
from xplorer_mini_sysid.lib.utils.controller import (create_position_controller, 
                                                    create_velocity_controller, 
                                                    Wrapper, 
                                                    PositionControllerType, 
                                                    VelocityControllerType
                                                    )


class CascadeKoopmanControl(Node):
    def __init__(self):
        super().__init__(node_name='cascade_kmpc', 
                         automatically_declare_parameters_from_overrides=True)
        
        self.get_logger().info('Cascade KMPC Node has been started.')

        # --- System Config ---
        self.is_vel_ready = False
        self.mlflow_uri = "https://mlflow.amarr.tan" 
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.mlflow_client = MlflowClient(tracking_uri=self.mlflow_uri)
        self.N_horizon = self.get_parameter('velocity_controller.params.N_horizon').value if self.get_parameter('velocity_controller.params.N_horizon') is not None else 10
        self.dt = 0.1

        # --- ROS Event ---
        self.add_on_set_parameters_callback(self.on_parameter_update)

        # self.create_subscription(AuvStatus, 'gnc/control_sync', self.control_callback, qos_profile_sensor_data)
        self.odom_sub = Subscriber(self, Odometry, 'gnc/odom_filtered', qos_profile=qos_profile_sensor_data)
        self.path_sub = Subscriber(self, Path, 'gnc/ref_trajectory/window')
        self.ts = ApproximateTimeSynchronizer([self.odom_sub, self.path_sub], queue_size=100, slop=0.15)
        self.ts.registerCallback(self.sync_callback)

        self.set_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.create_service(Trigger, 'gnc/koopman/sim_vel', self.sim_vel_callback)

        self.wrench_pub = self.create_publisher(WrenchStamped, 'gnc/cmd_wrench/wrench_desired', 10)
        self.twist_pub = self.create_publisher(TwistStamped, 'gnc/sysid/twist_desired', 10)
        self.twist_dot_pub = self.create_publisher(TwistStamped, 'gnc/sysid/eta_dot_ref', 10)
        self.err_twist_pub = self.create_publisher(TwistStamped, 'gnc/sysid/twist_error', 10)
        self.err_pose_pub = self.create_publisher(PoseStamped, 'gnc/sysid/pose_error', 10)

        # --- Declare controller type ---

        # Position controller
        self.pose_ctrl_type = self.get_parameter('position_controller.type').get_parameter_value().string_value

        if self.pose_ctrl_type == 'pi_ff':
            self.use_feedforward = self.get_parameter('position_controller.use_feedforward').get_parameter_value().bool_value
            self.use_filter = self.get_parameter('position_controller.use_filter').get_parameter_value().bool_value
        else:
            self.use_feedforward = False
            self.use_filter = False

        self.pose_ct: PositionControllerType = create_position_controller(**self.get_parameters_by_prefix('position_controller'))
        self.pose_ct_virtual: PositionControllerType =  create_position_controller(**self.get_parameters_by_prefix('position_controller'))

        # Velocity controller
        self.wrapper: Wrapper = self.load_model()    # load kopman model wrapper from MLflow
        self.vel_ctrl_type = self.get_parameter('velocity_controller.type').get_parameter_value().string_value

        if self.vel_ctrl_type == 'pid':
            self.vel_ct: VelocityControllerType = create_velocity_controller(logger=self.get_logger(),
                                                     **self.get_parameters_by_prefix('velocity_controller'))
            self.is_vel_ready = True
            self.get_logger().info("PID velocity controller selected, no model needed.")
        else: 
            self.use_preview = self.get_parameter('velocity_controller.use_preview').get_parameter_value().bool_value
            self.vel_ct: VelocityControllerType = create_velocity_controller(model=self.wrapper, 
                                                                            dt=self.dt,
                                                                            logger=self.get_logger(), 
                                                                            **self.get_parameters_by_prefix('velocity_controller'))
            self.is_vel_ready = True
            self.get_logger().info(f"{self.vel_ctrl_type} velocity controller created with model.")
        
        # --- State Variables ---
        # State vectors at current time step
        self.eta = np.zeros(6)
        self.nu = np.zeros(6)
        self.eta_error = np.zeros(6)
        self.eta_ref_prev = np.zeros(6)
        self.eta_dot_ref_filtered = np.zeros(6)
        self.alpha_ff = 0.2

        # Reference windows for cascade control (outer loop provides position reference, inner loop tracks velocity reference)
        self.nu_cmd_b = np.zeros(6)  

        # for sim trajectory
        self.control_mode = 1 # 0: IDLE, 1: NORMAL CONTROL, 2: SIM TRAJECTORY
        self.sim_index = 0              
        self.t_final = 60.0

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

    def load_model(self) -> Wrapper:
        name = self.get_parameter('model_name').value
        version = self.get_parameter('model_version').value
        try:
            model = self.mlflow_client.get_model_version(name, version)
            run = self.mlflow_client.get_run(model.run_id)
            loaded_model = mlflow.pyfunc.load_model(model.source).unwrap_python_model()
            self.get_logger().info(f"Successfully loaded model '{name}' from {run.info.run_name}.")
            return loaded_model

        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")

    def publish_pose_error(self, eta_error: np.ndarray):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.pose = pose_vect_to_msg(eta_error)
        self.err_pose_pub.publish(msg)

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

    def generate_virtual_reference_pid(self, eta_ref_window, eta, nu_cmd_b):
        
        # Preallocate 
        nu_hat_cmd_b_window = np.zeros((self.N_horizon, 6))
        eta_hat = np.copy(eta)
        nu_hat_cmd_b_window[0] = nu_cmd_b

        # Get current outer loop integral and derivative states for continuity in the virtual controller
        self.pose_ct_virtual.integral = np.copy(self.pose_ct.integral)
        self.pose_ct_virtual.prev_error = np.copy(self.pose_ct.prev_error)
        
        for i in range(1, self.N_horizon):
            eta_ref = eta_ref_window[i]
            nu_hat_cmd_b = self.pose_ct_virtual.compute_control(eta_hat, eta_ref, self.dt)
            J, _, _ = eulerang(eta_hat[3], eta_hat[4], eta_hat[5])
            nu_hat_cmd_b_window[i] = nu_hat_cmd_b
            eta_hat = eta_hat + self.dt * (J @ nu_hat_cmd_b)

        return nu_hat_cmd_b_window
    
    def generate_virtual_reference_ff_pi(self, eta_ref_window, eta, nu_cmd_b):
        # Numerical differentiation for Feed-forward (dot{eta}_ref)

        if self.use_filter:
            last_filt_val = np.copy(self.eta_dot_ref_filtered) 
            alpha = self.alpha_ff
            eta_dot_ref_window = np.zeros_like(eta_ref_window)
            for i in range(1, self.N_horizon):
                eta_dot_ref_raw_diff = cal_eta_err_with_ssa(eta_ref_window[i], eta_ref_window[i-1]) / self.dt
                eta_dot_ref_window[i] = (alpha * eta_dot_ref_raw_diff) + (1.0 - alpha) * last_filt_val
                last_filt_val = eta_dot_ref_window[i]
        else: 
            eta_dot_ref_window = np.gradient(eta_ref_window, self.dt, axis=0, edge_order=1)

        nu_hat_cmd_b_window = np.zeros((self.N_horizon, 6))
        
        # Clone current state for virtual reference simulation
        eta_hat = np.copy(eta)
        self.pose_ct_virtual.integral = np.copy(self.pose_ct.integral)
        
        # Initial step
        nu_hat_cmd_b_window[0] = nu_cmd_b

        for i in range(1, self.N_horizon):
            eta_ref = eta_ref_window[i]
            v_ff = eta_dot_ref_window[i] if self.use_feedforward else np.zeros(6)
            
            # Predict next velocity in Body Frame
            nu_b = self.pose_ct_virtual.compute_control(eta_hat, eta_ref, self.dt, v_ff)
            nu_hat_cmd_b_window[i] = nu_b
            
            # Kinematics Update: Convert Body Velocity to NED Velocity
            # dot{eta} = J(eta) * nu_b
            J, _, _ = eulerang(eta_hat[3], eta_hat[4], eta_hat[5]) # J in future
            # J, _, _ = eulerang(eta[3], eta[4], eta[5]) # J at current state for better stability in virtual sim
            eta_dot_hat = J @ nu_b
            
            # Update Virtual Pose (Forward Euler)
            eta_hat = eta_hat + (self.dt * eta_dot_hat)

        return nu_hat_cmd_b_window

    def on_parameter_update(self, params):

        for param in params:
            param_name = param.name
            
            if param_name.startswith('position_controller.'):
                if 'params' in param_name:
                    key = param_name.removeprefix('position_controller.params.')
                    val = param.value
                    self.pose_ct.set_params(**{key: val})
                    
                    # Structured Logging
                    self.get_logger().info(f"Position controller updated:\n{self.pose_ct.params.get(key)}")
                    if self.use_preview:
                        self.pose_ct_virtual.set_params(**{key: val})
                        self.get_logger().info(f"Virtual position controller parameters updated.\n {self.pose_ct_virtual.params.get(key)}")

            elif param_name.startswith('velocity_controller.'):
                if 'params' in param_name:
                    key = param_name.removeprefix('velocity_controller.params.')
                    val = param.value
                    if hasattr(self.vel_ct.params.weights, key):
                        val = np.diag(val)
                        self.vel_ct.set_params(**{key: val})
                        self.get_logger().info(f"Velocity controller updated:\n {np.array2string(getattr(self.vel_ct.params.weights, key), precision=2, suppress_small=True)}")

                    elif hasattr(self.vel_ct.params.bounds, key):
                        self.vel_ct.set_params(**{key: val})
                        self.get_logger().info(f"Velocity controller updated:\n {np.array2string(getattr(self.vel_ct.params.bounds, key), precision=2, suppress_small=True)}")
                    
            else:            
                self.get_logger().warning(f"Parameter update received for: {param_name}")

        return SetParametersResult(successful=True)

    def sync_callback(self, odom_msg, path_msg: Path):
        
        # get current state and reference from messages
        self.eta = odom_to_state_vect(odom_msg).flatten()[:6]
        self.nu = odom_to_state_vect(odom_msg).flatten()[6:12]
        eta_ref_window = np.zeros((self.N_horizon, 6))
        for i in range(len(path_msg.poses)):
            eta_ref_window[i, :] = pose_msg_to_vect(path_msg.poses[i].pose).flatten()
        
        # Position control 
        if self.use_feedforward:
            # eta_ref_dot_window = np.gradient(eta_ref_window, self.dt, axis=0, edge_order=1)    
            # eta_ref_dot = cal_eta_err_with_ssa(eta_ref_window[0], self.eta_ref_prev) / self.dt 
            
            if self.use_filter:
                eta_ref_dot_raw = cal_eta_err_with_ssa(eta_ref_window[0], self.eta_ref_prev) / self.dt  
                self.eta_dot_ref_filtered = (self.alpha_ff * eta_ref_dot_raw) + (1.0 - self.alpha_ff) * self.eta_dot_ref_filtered
                eta_ref_dot = self.eta_dot_ref_filtered
            else:
                eta_ref_dot = cal_eta_err_with_ssa(eta_ref_window[0], self.eta_ref_prev) / self.dt
            
            # Update previous reference for next iteration
            self.eta_ref_prev = eta_ref_window[0]

            # Compute body velocity command with feedforward
            nu_cmd_b = self.pose_ct.compute_control(self.eta, eta_ref_window[0], self.dt,  eta_ref_dot)
        else:
            # Compute body velocity command without feedforward
            nu_cmd_b = self.pose_ct.compute_control(self.eta, eta_ref_window[0], self.dt)

        # Velocity control
        if 'mpc' in self.vel_ctrl_type or 'lqr' in self.vel_ctrl_type:
            if self.use_preview:
                if self.pose_ctrl_type == 'pid':
                    self.nu_hat_cmd_b_window = self.generate_virtual_reference_pid(eta_ref_window, self.eta, nu_cmd_b)
                elif self.pose_ctrl_type == 'ff_pi':
                    self.nu_hat_cmd_b_window = self.generate_virtual_reference_ff_pi(eta_ref_window, self.eta, nu_cmd_b)
                self.tau_cmd = self.vel_ct.compute_control(self.nu, self.nu_hat_cmd_b_window)
            else:
                self.tau_cmd = self.vel_ct.compute_control(self.nu, nu_cmd_b)
        else:
            self.tau_cmd = self.vel_ct.compute_control(self.nu, nu_cmd_b, self.dt)
            
        # publish control commands
        if self.use_feedforward:
            self.publish_twist(self.twist_dot_pub, eta_ref_dot)  # Publish feedforward reference for debugging/analysis
        self.publish_wrench(self.tau_cmd)
        self.publish_pose_error(cal_eta_err_with_ssa(eta_ref_window[0], self.eta))
        self.publish_twist(self.twist_pub, nu_cmd_b)
        self.publish_twist(self.err_twist_pub, nu_cmd_b - self.nu)  

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
