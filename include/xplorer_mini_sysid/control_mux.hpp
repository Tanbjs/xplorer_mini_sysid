#ifndef __CONTROL_MUX_HPP__
#define __CONTROL_MUX_HPP_

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>

#include "xplorer_mini_sysid/signal_generate_lib.hpp"

class ControlMux : public rclcpp::Node 
{
public:
    ControlMux();
    ~ControlMux();

private:
    // Member variables for control modes
    std::string control_mode_;
    Eigen::Vector<double, 6> tau_desired_;

    // ROS Parameters
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
    rcl_interfaces::msg::SetParametersResult parametersCallback(const std::vector<rclcpp::Parameter> &parameters);
    void init_params_();

    // Timer callback
    rclcpp::TimerBase::SharedPtr timer_;
    void timer_callback_();

    // Declare publishers and callbacks
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_cmd_pub_;

    // Declare subscribers and callbacks
    rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr tau_desired_sub_;
    void tau_desired_sub_callback_(const geometry_msgs::msg::WrenchStamped::SharedPtr msg);

};

#endif // __CONTROL_MUX_HPP__