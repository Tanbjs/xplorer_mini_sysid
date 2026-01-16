#ifndef __CONTROL_MUX_HPP__
#define __CONTROL_MUX_HPP_

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>

#include "xplorer_mini_sysid/signal_generate_lib.hpp"

using namespace SysID;

class ControlMux : public rclcpp::Node 
{
public:
    ControlMux();
    ~ControlMux();

private:
    // Member variables for control modes and signal generation
    int signal_index_ = 0;
    int n_signals_ = 0;
    float dt_ = 0.1;
    float duration_ = 0.0;
    std::string control_mode_;

    Eigen::Vector<double, 6> tau_offset_ = {};
    Eigen::Vector<double, 6> tau_desired_ = {};
    Eigen::MatrixXd ext_signal_ = {};
    
    SignalGenerator::SignalType signal_type_;
    SignalGenerator::RBSConfig rbs_config_;
    SignalGenerator::RGSConfig rgs_config_;
    SignalGenerator::ChirpConfig chirp_config_;
    SignalGenerator::PRBSConfig prbs_config_;
    SignalGenerator::MultisineConfig multisine_config_;

    // ROS Parameters
    void init_params_();
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr params_callback_handle_;
    rcl_interfaces::msg::SetParametersResult update_params_(const std::vector<rclcpp::Parameter> &parameters);

    // Timer callback
    rclcpp::TimerBase::SharedPtr timer_;
    void timer_callback_();

    // Declare publishers and callbacks
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_cmd_pub_;

    // Declare subscribers and callbacks
    rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr tau_desired_sub_;
    void tau_desired_sub_callback_(const geometry_msgs::msg::WrenchStamped::SharedPtr msg);

    // Declare service servers and callbacks
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr signal_gen_trigger_srv_;
    void signal_gen_trigger_srv_callback_(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response);
};

#endif // __CONTROL_MUX_HPP__