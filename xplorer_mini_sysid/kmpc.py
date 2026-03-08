#!/usr/bin/env python3
import logging
import warnings
import os
import time
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

from xplorer_mini_sysid.mpc import controller
from xplorer_mini_sysid.mpc.base import MPCParams, Weights, Bounds
from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper

class CascadeKoopmanControl(Node):
    def __init__(self):
        super().__init__('gnc_control_cascade_kmpc')
        self.get_logger().info('Cascade KMPC Node has been started.')

        # --- 1. System Config ---
        self.is_vel_ready = False
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
            ('mpc_algorithm', 'standard'), 
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

        self.add_on_set_parameters_callback(self.parameters_callback)

        # Position controller
        self.pose_ct = PIDController(
            kp=np.array(self.get_parameter('Kp_pos').value),
            ki=np.array(self.get_parameter('Ki_pos').value),
            kd=np.array(self.get_parameter('Kd_pos').value),
            int_limit=np.array(self.get_parameter('integral_limit').value),
            max_vel=np.array(self.get_parameter('max_vel').value),
            logger=self.get_logger()
        )
                
        # Velocity controller
        self.wrapper = self.load_model()    # load kopman model wrapper from MLflow
        self.vel_ct = self.init_vel_ct()  # initialize MPC controller with loaded model

        # --- 5. ROS Interfaces ---
        self.create_subscription(AuvStatus, 'gnc/control_sync', self.control_callback, qos_profile_sensor_data)
        self.create_service(SetModel, 'gnc/koopman/set_model', self.set_model_callback)
        self.wrench_pub = self.create_publisher(WrenchStamped, 'gnc/cmd_wrench/wrench_desired', 10)
        self.twist_pub = self.create_publisher(TwistStamped, 'gnc/sysid/twist_desired', 10)
        self.err_twist_pub = self.create_publisher(TwistStamped, 'gnc/sysid/twist_error', 10)
    
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
                self.get_logger().info(f"Updated Q_diag: {np.diag(self.vel_ct.mpc_params.weights.Q)}")
            
            elif param.name == 'R_abs_diag':
                self.vel_ct.set_params(R_abs=np.diag(np.array(param.value)))
                self.get_logger().info(f"Updated R_abs_diag: {np.diag(self.vel_ct.mpc_params.weights.R_abs)}")
            
            elif param.name == 'R_rate_diag':
                self.vel_ct.set_params(R_rate=np.diag(np.array(param.value)))
                self.get_logger().info(f"Updated R_rate_diag: {np.diag(self.vel_ct.mpc_params.weights.R_rate)}")
            
            elif param.name == 'N_horizon':
                self.vel_ct.set_params(N_horizon=param.value)
                self.get_logger().info(f"Updated MPC horizon: {self.vel_ct.mpc_params.N_horizon}")
            
            elif param.name == 'max_tau':
                self.vel_ct.set_params(u_max=np.array(param.value))
                self.get_logger().info(f"Updated max control input limits: {self.vel_ct.mpc_params.bounds.u_max}")
            
            elif param.name == 'max_delta_tau':
                self.vel_ct.set_params(du_max=np.array(param.value))
                self.get_logger().info(f"Updated max change in control input limits: {self.vel_ct.mpc_params.bounds.du_max}")
            
            elif param.name == "rigid_body_mass":
                self.get_logger().info(f"Received new mass parameters: {param.value}")
                self.m_rb = np.array(param.value)
            
            elif param.name == "mpc_algorithm":
                self.get_logger().info(f"Switching MPC algorithm to: {param.value}")
                self.mpc_algorithm = param.value

            elif param.name == "include_abs_input":
                self.vel_ct._include_absolute_input = param.value
                self.get_logger().info(f"Updated include_abs_input flag: {self.vel_ct._include_absolute_input}")

        return SetParametersResult(successful=True)
    
    def control_callback(self, msg):
        eta = np.array(msg.eta)
        eta_error = np.array(msg.eta_e) 
        vel = np.array(msg.nu)

        t_start = time.time()

        if not self.is_vel_ready:
            self.get_logger().warn("Velocity controller not ready. Skipping control cycle.")
            return

        # NaN/Inf Guard
        if np.any(np.isnan(vel)) or np.any(np.isinf(vel)):
            self.get_logger().warn("NaN detected in velocity inputs. Skipping control cycle.")
            self.publish_wrench(np.zeros(6))
            self.publish_twist(self.twist_pub, np.zeros(6))
            return

        # 1. Outer Loop
        v_ref = self.pose_ct.compute_control(-eta_error, self.dt)
        J,_,_ = self.eulerang_nwu(eta[3], eta[4], eta[5])
        v_ref_body = np.linalg.inv(J) @ v_ref  # Tranvsform to body frame

        # 2. Inner Loop
        if self.get_parameter('mpc_algorithm').value == 'aug_error_output':
            tau_cmd = self.vel_ct.compute_control(vel, vel, v_ref_body)
        else:
            tau_cmd = self.vel_ct.compute_control(vel, v_ref_body)
        tau_cmd = np.asarray(tau_cmd).flatten()
        tau_cmd = np.clip(tau_cmd, -200, 200)

        # 3. Publish
        self.publish_wrench(tau_cmd)
        self.publish_twist(self.twist_pub, v_ref_body)
        self.publish_twist(self.err_twist_pub, v_ref_body - vel)

        t_exec = time.time() - t_start
        if t_exec > self.dt:
            self.get_logger().warn(f"Control loop overload! Exec time: {t_exec:.4f}s > {self.dt}s")        

    def load_model(self, version=None) -> DMDcWrapper | EDMDcWrapper | DeepModelWrapper:
        name = self.get_parameter('model_name').value

        # if version is not None:
        #     self.get_logger().info(f"Loading MLflow model: {name} (Version {version})")
        #     model_uri = f"models:/{name}/{version}"
        #     try:
        #         loaded_model = mlflow.pyfunc.load_model(model_uri)
        #         wrapper = loaded_model.unwrap_python_model() 
        #         self.vel_ct.setup_model(wrapper)
        #         self.get_logger().info("MPC Controller Initialized successfully.")
        #         return True
        #     except Exception as e:
        #         self.get_logger().error(f"Failed to load model version {version}: {e}")
        #         return False
            
        # else:
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
            return wrapper

        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")

    def set_model_callback(self, request, response):
        # Update parameter first
        param = rclpy.parameter.Parameter('model_name', rclpy.Parameter.Type.STRING, request.model_type)
        self.set_parameters([param])
        self.wrapper = self.load_model()
        response.success = False
        response.message = f"Failed to load model '{request.model_type}'"
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
    
    def init_vel_ct(self):
        if self.get_parameter('mpc_algorithm').value == 'standard':

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
            mpc_params = MPCParams(
                dt=self.dt,
                N_horizon=self.get_parameter('N_horizon').value,
                weights=weights,
                bounds=bounds
            )

            vel_ct = controller.Standard(model_wrapper=self.wrapper, mpc_params=mpc_params, node_name=self.get_name(), logger=self.get_logger())

        elif self.get_parameter('mpc_algorithm').value == 'aug_delayed_input':
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
            mpc_params = MPCParams(
                dt=self.dt,
                N_horizon=self.get_parameter('N_horizon').value,
                weights=weights,
                bounds=bounds
            )
            vel_ct = controller.AugmentDelayedInputForm(model_wrapper=self.wrapper, mpc_params=mpc_params, node_name=self.get_name(), logger=self.get_logger())
        
        elif self.get_parameter('mpc_algorithm').value == 'aug_error_output':
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
            mpc_params = MPCParams(
                dt=self.dt,
                N_horizon=self.get_parameter('N_horizon').value,
                weights=weights,
                bounds=bounds
            )
            vel_ct = controller.AugmentErrorOutputForm(model_wrapper=self.wrapper, mpc_params=mpc_params, node_name=self.get_name(), logger=self.get_logger())
        
        elif self.get_parameter('mpc_algorithm').value == 'implicit_rigid_tube':
            pass # Placeholder for tube MPC initialization, can be implemented similarly with additional tube parameters

        else:
            raise ValueError(f"Unsupported MPC algorithm: {self.get_parameter('mpc_algorithm').value}")
        
        self.is_vel_ready = True
        return vel_ct


class PIDController:
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