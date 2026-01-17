#include "xplorer_mini_sysid/control_mux.hpp"

ControlMux::ControlMux() : Node("control_mux") 
{
    // Initialize parameters
    init_params_();
    params_callback_handle_ = this->add_on_set_parameters_callback(
        std::bind(&ControlMux::update_params_, this, std::placeholders::_1)
    );

    // Initialize publisher
    wrench_cmd_pub_ = this->create_publisher<geometry_msgs::msg::WrenchStamped>("gnc/cmd_wrench/control", 10);
    wrench_noise_pub_ = this->create_publisher<geometry_msgs::msg::WrenchStamped>("gnc/cmd_wrench/wrench_noise", 10);

    // Initialize subscriber
    tau_desired_sub_ = this->create_subscription<geometry_msgs::msg::WrenchStamped>("gnc/cmd_wrench/wrench_desired", 10, 
        std::bind(&ControlMux::tau_desired_sub_callback_, this, std::placeholders::_1));
    
    // Initialize service server
    signal_gen_trigger_srv_ = this->create_service<std_srvs::srv::Trigger>("gnc/control_mux/signal_gen_trigger", 
        std::bind(&ControlMux::signal_gen_trigger_srv_callback_, this, std::placeholders::_1, std::placeholders::_2));

    // Initialize timer
    timer_ = this->create_wall_timer(std::chrono::duration<float>(dt_),
        std::bind(&ControlMux::timer_callback_, this));  
}

void ControlMux::init_params_() 
{
    // Initialize ROS parameters with default values
    dt_ = this->declare_parameter("dt", 0.1);
    enable_offset_ = this->declare_parameter("enable_offset", false);
    control_mode_ = ControlMux::stringToControlMode(this->declare_parameter("control_mode", "closed"));
    signal_type_ = SignalGenerator::stringToSignalType(this->declare_parameter("signal_type", "RBS"));
    duration_ = this->declare_parameter("duration", 0.0);
    n_signals_ = this->declare_parameter("n_signals", 0);
    wrench_offset_ = Eigen::Map<const Eigen::Vector<double, 6>>(this->declare_parameter("wrench_offset", std::vector<double>(6, 0.0)).data());
    
    rbs_config_.min = Eigen::Map<const Eigen::Vector<double, 6>>(this->declare_parameter("rbs_config.min", std::vector<double>(6, 0.0)).data());
    rbs_config_.max = Eigen::Map<const Eigen::Vector<double, 6>>(this->declare_parameter("rbs_config.max", std::vector<double>(6, 0.0)).data());

    rgs_config_.mean = Eigen::Map<const Eigen::Vector<double, 6>>(this->declare_parameter("rgs_config.mean", std::vector<double>(6, 0.0)).data());
    rgs_config_.stddev = Eigen::Map<const Eigen::Vector<double, 6>>(this->declare_parameter("rgs_config.stddev", std::vector<double>(6, 0.0)).data());

    multisine_config_.amplitudes = Eigen::Map<const Eigen::Vector<double, 6>>(this->declare_parameter("multisine_config.amplitudes", std::vector<double>(6, 0.0)).data());
    multisine_config_.grid_skips = this->declare_parameter("multisine_config.grid_skips", 0);
    multisine_config_.n_trails = this->declare_parameter("multisine_config.n_trails", 0);
    multisine_config_.n_sines = this->declare_parameter("multisine_config.n_sines", 0);

    // Log initialized parameters
    RCLCPP_INFO(this->get_logger(), "############## Initialized parameters ##############");
    RCLCPP_INFO(this->get_logger(), "### General parameters ###");
    RCLCPP_INFO(this->get_logger(), "- dt_: %f", dt_);
    RCLCPP_INFO(this->get_logger(), "- control_mode_: %s", this->get_parameter("control_mode").as_string().c_str());
    RCLCPP_INFO(this->get_logger(), "- enable_offset_: %s", enable_offset_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "- signal_type_: %s", this->get_parameter("signal_type").as_string().c_str());
    RCLCPP_INFO(this->get_logger(), "### General signal parameters ###");
    RCLCPP_INFO(this->get_logger(), "- duration_: %f", duration_);
    RCLCPP_INFO(this->get_logger(), "- n_signals_: %d", n_signals_);
    RCLCPP_INFO_STREAM(this->get_logger(), "- wrench_offset_: " << wrench_offset_.transpose());
    RCLCPP_INFO(this->get_logger(), "### Random binary sequence generator parameters ###");
    RCLCPP_INFO_STREAM(this->get_logger(), "- rbs_config_.min: " << rbs_config_.min.transpose());
    RCLCPP_INFO_STREAM(this->get_logger(), "- rbs_config_.max: " << rbs_config_.max.transpose());
    RCLCPP_INFO(this->get_logger(), "### Random Gaussian sequence generator parameters ###");
    RCLCPP_INFO_STREAM(this->get_logger(), "- rgs_config_.mean: " << rgs_config_.mean.transpose());
    RCLCPP_INFO_STREAM(this->get_logger(), "- rgs_config_.stddev: " << rgs_config_.stddev.transpose());
    RCLCPP_INFO(this->get_logger(), "### Multisine signal generator parameters ###");
    RCLCPP_INFO_STREAM(this->get_logger(), "- multisine_config_.amplitudes: " << multisine_config_.amplitudes.transpose());
    RCLCPP_INFO(this->get_logger(), "- multisine_config_.grid_skips: %d", multisine_config_.grid_skips);
    RCLCPP_INFO(this->get_logger(), "- multisine_config_.n_trails: %d", multisine_config_.n_trails);
    RCLCPP_INFO(this->get_logger(), "- multisine_config_.n_sines: %d", multisine_config_.n_sines);
    RCLCPP_INFO(this->get_logger(), "###################################################");
}

rcl_interfaces::msg::SetParametersResult ControlMux::update_params_(const std::vector<rclcpp::Parameter> &parameters) 
{
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    for (const auto &param : parameters) 
    {
        if (param.get_name() == "dt") 
        {
            dt_ = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated dt_: %f", dt_);
        }
        else if (param.get_name() == "control_mode") 
        {
            control_mode_ = ControlMux::stringToControlMode(param.as_string());
            RCLCPP_INFO(this->get_logger(), "Updated control_mode_: %s", this->get_parameter("control_mode").as_string().c_str());
        }
        else if (param.get_name() == "enable_offset") 
        {
            enable_offset_ = param.as_bool();
            RCLCPP_INFO(this->get_logger(), "Updated enable_offset_: %s", enable_offset_ ? "true" : "false");
        } 
        else if (param.get_name() == "signal_type") 
        {
            signal_type_ = SignalGenerator::stringToSignalType(param.as_string());
            RCLCPP_INFO(this->get_logger(), "Updated signal_type_: %s", this->get_parameter("signal_type").as_string().c_str());
        }
        else if (param.get_name() == "duration") 
        {
            duration_ = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated duration_: %f", duration_);
        } 
        else if (param.get_name() == "n_signals") 
        {
            n_signals_ = param.as_int();
            RCLCPP_INFO(this->get_logger(), "Updated n_signals_: %d", n_signals_);
        } 
        else if (param.get_name() == "wrench_offset") 
        {
            wrench_offset_ = Eigen::Map<const Eigen::Vector<double, 6>>(param.as_double_array().data());
            RCLCPP_INFO_STREAM(this->get_logger(), "Updated wrench_offset_: " << wrench_offset_.transpose());
        } 
        else if (param.get_name() == "rbs_config.min")
        {
            rbs_config_.min = Eigen::Map<const Eigen::Vector<double, 6>>(param.as_double_array().data());
            RCLCPP_INFO_STREAM(this->get_logger(), "Updated rbs_config_.min: " << rbs_config_.min.transpose());
        } 
        else if (param.get_name() == "rbs_config.max") 
        {
            rbs_config_.max = Eigen::Map<const Eigen::Vector<double, 6>>(param.as_double_array().data());
            RCLCPP_INFO_STREAM(this->get_logger(), "Updated rbs_config_.max: " << rbs_config_.max.transpose());
        } 
        else if (param.get_name() == "rgs_config.mean") 
        {
            rgs_config_.mean = Eigen::Map<const Eigen::Vector<double, 6>>(param.as_double_array().data());
            RCLCPP_INFO_STREAM(this->get_logger(), "Updated rgs_config_.mean: " << rgs_config_.mean.transpose());
        }
        else if (param.get_name() == "rgs_config.stddev") 
        {
            rgs_config_.stddev = Eigen::Map<const Eigen::Vector<double, 6>>(param.as_double_array().data());    
            RCLCPP_INFO_STREAM(this->get_logger(), "Updated rgs_config_.stddev: " << rgs_config_.stddev.transpose());
        } 
        else if (param.get_name() == "multisine_config.amplitudes")
        {
            multisine_config_.amplitudes = Eigen::Map<const Eigen::Vector<double, 6>>(param.as_double_array().data());
            RCLCPP_INFO_STREAM(this->get_logger(), "Updated multisine_config_.amplitudes: " << multisine_config_.amplitudes.transpose());
        } 
        else if (param.get_name() == "multisine_config.grid_skips") 
        {
            multisine_config_.grid_skips = param.as_int();
            RCLCPP_INFO(this->get_logger(), "Updated multisine_config_.grid_skips: %d", multisine_config_.grid_skips);
        } 
        else if (param.get_name() == "multisine_config.n_trails") 
        {
            multisine_config_.n_trails = param.as_int();
            RCLCPP_INFO(this->get_logger(), "Updated multisine_config_.n_trails: %d", multisine_config_.n_trails);
        } 
        else if (param.get_name() == "multisine_config.n_sines") 
        {
            multisine_config_.n_sines = param.as_int();
            RCLCPP_INFO(this->get_logger(), "Updated multisine_config_.n_sines: %d", multisine_config_.n_sines);
        }
    }
    return result;
}

void ControlMux::tau_desired_sub_callback_(const geometry_msgs::msg::WrenchStamped::SharedPtr msg) 
{
    // Extract desired torques from the message
    wrench_desired_(0) = msg->wrench.force.x;
    wrench_desired_(1) = msg->wrench.force.y;
    wrench_desired_(2) = msg->wrench.force.z;
    wrench_desired_(3) = msg->wrench.torque.x;
    wrench_desired_(4) = msg->wrench.torque.y;
    wrench_desired_(5) = msg->wrench.torque.z;
};

void ControlMux::timer_callback_() 
{
    wrench_msg.header.stamp = this->now();
    wrench_noise_msg.header.stamp = this->now();

    switch (control_mode_) 
    {
        case ControlMode::OPEN_LOOP:
            if (is_signal_gen_active_) 
            {
                wrench_cmd_ = ext_signal_.row(signal_index_).transpose();
                signal_index_++;
            } 
            else 
            {
                wrench_cmd_.setZero();
            }
            break;

        case ControlMode::CLOSED_LOOP:
            wrench_cmd_ = wrench_desired_;
            if (is_signal_gen_active_) 
            {
                wrench_cmd_ += ext_signal_.row(signal_index_).transpose();
                signal_index_++;
            }
            else 
            {
                wrench_cmd_ = wrench_desired_;
            }
            break;

        default:
            RCLCPP_ERROR(this->get_logger(), "Unknown control mode!");
            wrench_cmd_.setZero();
            break;
    }

    // Add offset to last commanded torques if enabled
    if (enable_offset_) {
        wrench_cmd_ += wrench_offset_;
    }

    // Final wrench command
    wrench_msg.wrench.force.x = wrench_cmd_(0);
    wrench_msg.wrench.force.y = wrench_cmd_(1);
    wrench_msg.wrench.force.z = wrench_cmd_(2);
    wrench_msg.wrench.torque.x = wrench_cmd_(3);
    wrench_msg.wrench.torque.y = wrench_cmd_(4);
    wrench_msg.wrench.torque.z = wrench_cmd_(5);
    
    // Publish messages
    wrench_noise_pub_->publish(wrench_noise_msg);
    wrench_cmd_pub_->publish(wrench_msg);

    // Check if signal generation is completed
    if (is_signal_gen_active_ && signal_index_ >= ext_signal_.rows()) 
    {
        is_signal_gen_active_ = false;
        signal_index_ = 0;
        RCLCPP_INFO(this->get_logger(), "Signal generation completed.");
    }
}

void ControlMux::signal_gen_trigger_srv_callback_(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response) 
{
    RCLCPP_INFO(this->get_logger(), "Signal generation triggered via service call.");
    int n_samples = static_cast<int>(duration_ / dt_);
    is_signal_gen_active_ = true;

    switch (signal_type_)
    {
    case SignalGenerator::SignalType::RBS:
        ext_signal_ = SignalGenerator::RBS(n_samples, n_signals_, rbs_config_);
        RCLCPP_INFO(this->get_logger(), "############### Generated RBS signal ###############");
        RCLCPP_INFO_STREAM(this->get_logger(), "- max: " << rbs_config_.max.transpose());
        RCLCPP_INFO_STREAM(this->get_logger(), "- min: " << rbs_config_.min.transpose());
        break;
    
    case SignalGenerator::SignalType::RGS:
        ext_signal_ = SignalGenerator::RGS(n_samples, n_signals_, rgs_config_);
        RCLCPP_INFO(this->get_logger(), "############### Generated RGS signal ###############");
        RCLCPP_INFO_STREAM(this->get_logger(), "- mean: " << rgs_config_.mean.transpose());
        RCLCPP_INFO_STREAM(this->get_logger(), "- stddev: " << rgs_config_.stddev.transpose());
        break;
    
    case SignalGenerator::SignalType::MULTISINE:
        ext_signal_ = SignalGenerator::Multisine(n_samples, n_signals_, multisine_config_);
        RCLCPP_INFO(this->get_logger(), "############### Generated Multisine signal ###############");
        RCLCPP_INFO(this->get_logger(), "- n_sines: %d", multisine_config_.n_sines);
        RCLCPP_INFO(this->get_logger(), "- grid_skips: %d", multisine_config_.grid_skips);
        RCLCPP_INFO(this->get_logger(), "- n_trails: %d", multisine_config_.n_trails);
        RCLCPP_INFO_STREAM(this->get_logger(), "- amplitudes: " << multisine_config_.amplitudes.transpose());
        break;
    
    case SignalGenerator::SignalType::CHIRP:
        ext_signal_ = SignalGenerator::Chirp(n_samples, n_signals_, chirp_config_);
        RCLCPP_INFO(this->get_logger(), "Generated Chirp signal");
        break;
    
    case SignalGenerator::SignalType::PRBS:
        ext_signal_ = SignalGenerator::PRBS(n_samples, n_signals_, prbs_config_);
        RCLCPP_INFO(this->get_logger(), "Generated PRBS signal");
        break;
    
    default:
        RCLCPP_ERROR(this->get_logger(), "Unknown signal type!");
        response->success = false;
        response->message = "Unknown signal type.";
        return;
    }

    response->success = true;
    response->message = "Signal generation started.";
}

ControlMux::~ControlMux() 
{
    // Destructor implementation (if needed)
}

int main(int argc, char **argv) 
{
    rclcpp::init(argc, argv);
    auto control_mux_node = std::make_shared<ControlMux>();
    rclcpp::spin(control_mux_node);
    rclcpp::shutdown();
}