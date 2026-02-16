#!/usr/bin/env python3

import numpy as np
import mlflow
import rclpy
from rclpy.node import Node
import message_filters

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, WrenchStamped

class KMMonitor(Node):
    def __init__(self):
        super().__init__('km_monitor')
        self.get_logger().info("KM One-step Monitor Node started.")

        # --- MLflow Configuration ---
        # Tracking server URI (hosted on NAS)
        self.mlflow_uri = "https://mlflow.amarr.tan"
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.client = mlflow.tracking.MlflowClient(tracking_uri=self.mlflow_uri)

        # --- System Parameters & State Variables ---
        self.u_timeout = 0.5        # Fallback to zero input if wrench msg is delayed
        self.last_wrench_time = self.get_clock().now()
        
        self.nu_true = np.zeros(6)  # Current velocity state
        self.u_true = np.zeros(6)   # Current control input (wrench)

        # --- Models Initialization (Load latest from Registry) ---
        self.dmdc_model = self.init_model("dmdc")
        self.edmdc_model = self.init_model("edmdc")
        self.nn_model = self.init_model("deep vanilla")

        # --- Publishers (One-step prediction output) ---
        self.pub_dmdc = self.create_publisher(TwistStamped, "gnc/sysid/dmdc/one_step", 10)
        self.pub_edmdc = self.create_publisher(TwistStamped, "gnc/sysid/edmdc/one_step", 10)
        self.pub_nn = self.create_publisher(TwistStamped, "gnc/sysid/nn/one_step", 10)

        # --- Subscribers & Message Synchronization ---
        # Synchronize Odom and Wrench to ensure state-action pairs (x, u) are time-aligned
        self.odom_sub = message_filters.Subscriber(self, Odometry, "gnc/odom_filtered")
        self.wrench_sub = message_filters.Subscriber(self, WrenchStamped, "gnc/est_tau")
        
        # ApproximateTimeSynchronizer allows small timestamp differences (slop)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.wrench_sub], 
            queue_size=10, 
            slop=0.05
        )
        self.ts.registerCallback(self.sync_callback)

        # Fallback subscriber: triggered every odom update to check for wrench timeouts
        self.odom_fallback_sub = self.create_subscription(
            Odometry, "gnc/odom_filtered", self.odom_fallback_callback, 10
        )

    def init_model(self, model_name):
        """Fetches the latest version from MLflow Model Registry and unwraps it."""
        try:
            versions = self.client.get_latest_versions(model_name, stages=["None"])
            latest = versions[0].version
            model_uri = f"models:/{model_name}/{latest}"
            loaded = mlflow.pyfunc.load_model(model_uri)
            self.get_logger().info(f"Successfully loaded {model_name} version {latest}")
            # Unwrap to access specific Koopman methods if needed
            return loaded.unwrap_python_model()
        except Exception as e:
            self.get_logger().error(f"Failed to load model '{model_name}': {e}")
            return None

    def sync_callback(self, odom_msg, wrench_msg):
        """Callback for time-aligned state and control input."""
        self.last_wrench_time = self.get_clock().now()
        u = self.extract_wrench(wrench_msg)
        nu = self.extract_odom(odom_msg)
        self.process_inference(nu, u)

    def odom_fallback_callback(self, odom_msg):
        """Handles cases where Wrench data is missing or publisher is dead."""
        now = self.get_clock().now()
        dt_since_last_u = (now - self.last_wrench_time).nanoseconds / 1e9
        
        if dt_since_last_u > self.u_timeout:
            # If wrench times out, assume unforced system (u=0)
            nu = self.extract_odom(odom_msg)
            self.process_inference(nu, np.zeros(6))

    def extract_odom(self, msg):
        """Converts Odometry twist into a 1x6 numpy array."""
        return np.array([
            msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z,
            msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z
        ])

    def extract_wrench(self, msg):
        """Converts WrenchStamped into a 1x6 numpy array."""
        return np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        ])

    def process_inference(self, nu, u):
        """Prepares input features and executes model predictions."""
        # Input vector format: [velocity_states, control_inputs]
        # Execute and publish predictions for enabled models
        if self.dmdc_model:
            self.publish_twist(self.pub_dmdc, self.dmdc_model.predict(context=None, model_input={'x': nu, 'u': u}))

        if self.edmdc_model:
            self.publish_twist(self.pub_edmdc, self.edmdc_model.predict(context=None, model_input={'x': nu, 'u': u}))

        if self.nn_model:
            self.publish_twist(self.pub_nn, self.nn_model.predict(context=None, model_input={'x': nu, 'u': u}))

    def publish_twist(self, pub, data):
        """Helper to publish numpy array as TwistStamped message."""
        val = data.flatten()
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        
        # Mapping: [u, v, w, p, q, r]
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = val[0:3]
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = val[3:6]
        pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = KMMonitor()
    try:
        # Use MultiThreadedExecutor to handle concurrent callbacks without blocking
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()