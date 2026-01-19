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
    // Control modes
    enum class ControlMode 
    {
        OPEN_LOOP,
        CLOSED_LOOP,
    };
    
    inline ControlMode stringToControlMode(const std::string &mode_str) 
    {
        if (mode_str == "open") return ControlMode::OPEN_LOOP;
        if (mode_str == "closed") return ControlMode::CLOSED_LOOP;
    }
    
    // Member variables for control modes and signal generation
    // member variables
    bool is_signal_gen_active_ = false;
    bool enable_offset_ = false;
    int signal_index_ = 0;
    int n_signals_ = 0;
    float dt_ = 0.1;
    float duration_ = 0.0;
    ControlMode control_mode_ = ControlMode::CLOSED_LOOP;

    Eigen::Vector<double, 6> wrench_desired_ = Eigen::VectorXd::Zero(6);
    Eigen::Vector<double, 6> wrench_offset_ = Eigen::VectorXd::Zero(6);
    Eigen::Vector<double, 6> wrench_noise_ = Eigen::VectorXd::Zero(6);
    Eigen::Vector<double, 6> wrench_cmd_ = Eigen::VectorXd::Zero(6);
    Eigen::MatrixXd ext_signal_ = Eigen::MatrixXd::Zero(0, 0);
    
    SignalGenerator::SignalType signal_type_;
    SignalGenerator::RBSConfig rbs_config_;
    SignalGenerator::RGSConfig rgs_config_;
    SignalGenerator::ChirpConfig chirp_config_;
    SignalGenerator::PRBSConfig prbs_config_;
    SignalGenerator::MultisineConfig multisine_config_;

    // member functions
    std::tuple<bool, Eigen::MatrixXd> generate_signal_(int n_samples, int n_signals, SignalGenerator::SignalType signal_type);

    // ROS Parameters
    void init_params_();
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr params_callback_handle_;
    rcl_interfaces::msg::SetParametersResult update_params_(const std::vector<rclcpp::Parameter> &parameters);

    // Timer callback
    rclcpp::TimerBase::SharedPtr timer_;
    void timer_callback_();

    // Declare publishers and callbacks
    geometry_msgs::msg::WrenchStamped wrench_msg;
    geometry_msgs::msg::WrenchStamped wrench_noise_msg;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_cmd_pub_;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_noise_pub_;

    // Declare subscribers and callbacks
    rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr tau_desired_sub_;
    void tau_desired_sub_callback_(const geometry_msgs::msg::WrenchStamped::SharedPtr msg);

    // Declare service servers and callbacks
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr signal_gen_trigger_srv_;
    void signal_gen_trigger_srv_callback_(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr inject_control_noise_srv_;
    void inject_control_noise_srv_callback_(
        const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
        std::shared_ptr<std_srvs::srv::SetBool::Response> response);
    rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr record_client_;
    std::shared_ptr<std_srvs::srv::SetBool::Request> record_request_;
};

#endif // __CONTROL_MUX_HPP__