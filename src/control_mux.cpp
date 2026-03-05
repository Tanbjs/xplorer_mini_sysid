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
    ocean_current_pub_ = this->create_publisher<geometry_msgs::msg::WrenchStamped>("gnc/cmd_wrench/ocean_current", 10);

    // Initialize subscriber
    tau_desired_sub_ = this->create_subscription<geometry_msgs::msg::WrenchStamped>("gnc/cmd_wrench/wrench_desired", 10, 
        std::bind(&ControlMux::tau_desired_sub_callback_, this, std::placeholders::_1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("gnc/odom_filtered", 10, 
        std::bind(&ControlMux::odom_sub_callback_, this, std::placeholders::_1));
    
    // Initialize service server
    signal_gen_trigger_srv_ = this->create_service<std_srvs::srv::Trigger>("gnc/control_mux/signal_gen_trigger", 
        std::bind(&ControlMux::signal_gen_trigger_srv_callback_, this, std::placeholders::_1, std::placeholders::_2));
    inject_control_noise_srv_ = this->create_service<std_srvs::srv::SetBool>("gnc/control_mux/inject_control_noise", 
        std::bind(&ControlMux::inject_control_noise_srv_callback_, this, std::placeholders::_1, std::placeholders::_2));
    record_client_ = this->create_client<std_srvs::srv::SetBool>("gnc/record");
    record_request_ = std::make_shared<std_srvs::srv::SetBool::Request>();

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

    // ocean current parameters
    oc_config_.enable = this->declare_parameter("ocean_current.enable", false);
    oc_config_.seed = this->declare_parameter("ocean_current.seed", 255);

    oc_config_.Vc.mu = this->declare_parameter("ocean_current.Vc.mu", 0.0);
    oc_config_.Vc.mean_w = this->declare_parameter("ocean_current.Vc.mean_w", 0.0);
    oc_config_.Vc.var_w = this->declare_parameter("ocean_current.Vc.var_w", 0.01);
    oc_config_.Vc.init_val = this->declare_parameter("ocean_current.Vc.init_val", 0.0);
    oc_config_.Vc.min_val = this->declare_parameter("ocean_current.Vc.min_val", -1.5);
    oc_config_.Vc.max_val = this->declare_parameter("ocean_current.Vc.max_val", 1.5);

    oc_config_.alpha.mu = this->declare_parameter("ocean_current.alpha.mu", 0.0);
    oc_config_.alpha.mean_w = this->declare_parameter("ocean_current.alpha.mean_w", 0.0);
    oc_config_.alpha.var_w = this->declare_parameter("ocean_current.alpha.var_w", 0.00060923);
    oc_config_.alpha.init_val = this->declare_parameter("ocean_current.alpha.init_val", 0.0);
    oc_config_.alpha.min_val = this->declare_parameter("ocean_current.alpha.min_val", -M_PI);
    oc_config_.alpha.max_val = this->declare_parameter("ocean_current.alpha.max_val", M_PI);

    oc_config_.beta.mu = this->declare_parameter("ocean_current.beta.mu", 0.0);
    oc_config_.beta.mean_w = this->declare_parameter("ocean_current.beta.mean_w", 0.0);
    oc_config_.beta.var_w = this->declare_parameter("ocean_current.beta.var_w", 0.00060923);
    oc_config_.beta.init_val = this->declare_parameter("ocean_current.beta.init_val", 0.0);
    oc_config_.beta.min_val = this->declare_parameter("ocean_current.beta.min_val", -M_PI);
    oc_config_.beta.max_val = this->declare_parameter("ocean_current.beta.max_val", M_PI);

    // Dynamic Parameters
    this->declare_parameter("fluid_dense", 1024.0);
    auv_dyn.fluid_dense = this->get_parameter("fluid_dense").as_double();

    this->declare_parameter("gravity", 9.81);
    auv_dyn.gravity = this->get_parameter("gravity").as_double();

    this->declare_parameter("weight", 0.0);
    auv_dyn.weight = this->get_parameter("weight").as_double();

    this->declare_parameter("buoyancy", 0.0);
    auv_dyn.buoyancy = this->get_parameter("buoyancy").as_double();

    std::vector<double> my_double_array;

    this->declare_parameter("r_g", std::vector<double>(3, 0.0));
    my_double_array = this->get_parameter("r_g").as_double_array();
    auv_dyn.r_g = Eigen::Map<Eigen::Matrix<double, 3, 1>>(my_double_array.data());
    
    this->declare_parameter("r_b", std::vector<double>(3, 0.0));
    my_double_array = this->get_parameter("r_b").as_double_array();
    auv_dyn.r_b = Eigen::Map<Eigen::Matrix<double, 3, 1>>(my_double_array.data());
    
    this->declare_parameter("rigid_body_mass", std::vector<double>(36, 0.0));
    my_double_array = this->get_parameter("rigid_body_mass").as_double_array();
    auv_dyn.m_rb_mat = Eigen::Map<Eigen::Matrix<double, 6, 6>>(my_double_array.data());
    
    this->declare_parameter("added_mass", std::vector<double>(36, 0.0));
    my_double_array = this->get_parameter("added_mass").as_double_array();
    auv_dyn.m_a_mat = Eigen::Map<Eigen::Matrix<double, 6, 6>>(my_double_array.data());

    this->declare_parameter("linear_drag", std::vector<double>(36, 0.0));
    my_double_array = this->get_parameter("linear_drag").as_double_array();
    auv_dyn.d_l_mat = Eigen::Map<Eigen::Matrix<double, 6, 6>>(my_double_array.data());
    
    this->declare_parameter("nonlinear_drag", std::vector<double>(36, 0.0));
    my_double_array = this->get_parameter("nonlinear_drag").as_double_array();
    auv_dyn.d_nl_mat = Eigen::Map<Eigen::Matrix<double, 6, 6>>(my_double_array.data());
    
    this->declare_parameter("thruster_map", std::vector<double>(48, 0.0));
    my_double_array = this->get_parameter("thruster_map").as_double_array();
    auv_dyn.thrust_map_mat = Eigen::Map<Eigen::Matrix<double, 6, 8>>(my_double_array.data());

    // Log initialized parameters
    RCLCPP_INFO(this->get_logger(), "############## Initialized parameters ##############");
    RCLCPP_INFO(this->get_logger(), "### General parameters ###");
    RCLCPP_INFO(this->get_logger(), "- dt_: %f", dt_);
    RCLCPP_INFO(this->get_logger(), "- control_mode_: %s", this->get_parameter("control_mode").as_string().c_str());
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
        else if (param.get_name() == "ocean_current.enable")
        {
            oc_config_.enable = param.as_bool();
            if (oc_config_.enable) {
                oc_generator_ = std::make_unique<OceanEnvironment::OceanCurrentGenerator>(dt_, oc_config_, oc_config_.seed);
            }
            RCLCPP_INFO(this->get_logger(), "Updated ocean_current.enable: %s", oc_config_.enable ? "true" : "false");
        }
        else if (param.get_name() == "ocean_current.seed")
        {
            oc_config_.seed = param.as_int();
            RCLCPP_INFO(this->get_logger(), "Updated ocean_current.seed: %d", oc_config_.seed);
        }
        else if (param.get_name() == "ocean_current.Vc.mu") 
        {
            oc_config_.Vc.mu = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated ocean_current.Vc.mu: %f", oc_config_.Vc.mu);
        }
        else if (param.get_name() == "ocean_current.Vc.mean_w") 
        {
            oc_config_.Vc.mean_w = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated ocean_current.Vc.mean_w: %f", oc_config_.Vc.mean_w);
        }
        else if (param.get_name() == "ocean_current.Vc.var_w") 
        {
            oc_config_.Vc.var_w = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated ocean_current.Vc.var_w: %f", oc_config_.Vc.var_w);
        }
        else if (param.get_name() == "ocean_current.Vc.init_val") 
        {
            oc_config_.Vc.init_val = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated ocean_current.Vc.init_val: %f", oc_config_.Vc.init_val);
        }
        else if (param.get_name() == "ocean_current.Vc.min_val") 
        {
            oc_config_.Vc.min_val = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated ocean_current.Vc.min_val: %f", oc_config_.Vc.min_val);
        }
        else if (param.get_name() == "ocean_current.Vc.max_val") 
        {
            oc_config_.Vc.max_val = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated ocean_current.Vc.max_val: %f", oc_config_.Vc.max_val);
        }
        else if (param.get_name() == "ocean_current.alpha.mu") 
        {
            oc_config_.alpha.mu = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated ocean_current.alpha.mu: %f", oc_config_.alpha.mu);
        }
        else if (param.get_name() == "ocean_current.alpha.mean_w") 
        {
            oc_config_.alpha.mean_w = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated ocean_current.alpha.mean_w: %f", oc_config_.alpha.mean_w);
        }
        else if (param.get_name() == "ocean_current.alpha.var_w") 
        {
            oc_config_.alpha.var_w = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated ocean_current.alpha.var_w: %f", oc_config_.alpha.var_w);
        }
        else if (param.get_name() == "ocean_current.alpha.init_val") 
        {
            oc_config_.alpha.init_val = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated ocean_current.alpha.init_val: %f", oc_config_.alpha.init_val);
        }
        else if (param.get_name() == "ocean_current.alpha.min_val") 
        {
            oc_config_.alpha.min_val = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated ocean_current.alpha.min_val: %f", oc_config_.alpha.min_val);
        }
        else if (param.get_name() == "ocean_current.alpha.max_val") 
        {
            oc_config_.alpha.max_val = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated ocean_current.alpha.max_val: %f", oc_config_.alpha.max_val);
        }
        else if (param.get_name() == "ocean_current.beta.mu") 
        {
            oc_config_.beta.mu = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated ocean_current.beta.mu: %f", oc_config_.beta.mu);
        }
        else if (param.get_name() == "ocean_current.beta.mean_w") 
        {
            oc_config_.beta.mean_w = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated ocean_current.beta.mean_w: %f", oc_config_.beta.mean_w);
        }
        else if (param.get_name() == "ocean_current.beta.var_w") 
        {
            oc_config_.beta.var_w = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated ocean_current.beta.var_w: %f", oc_config_.beta.var_w);
        }
        else if (param.get_name() == "ocean_current.beta.init_val") 
        {
            oc_config_.beta.init_val = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated ocean_current.beta.init_val: %f", oc_config_.beta.init_val);
        }
        else if (param.get_name() == "ocean_current.beta.min_val") 
        {
            oc_config_.beta.min_val = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated ocean_current.beta.min_val: %f", oc_config_.beta.min_val);
        }
        else if (param.get_name() == "ocean_current.beta.max_val") 
        {
            oc_config_.beta.max_val = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated ocean_current.beta.max_val: %f", oc_config_.beta.max_val);
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

void ControlMux::odom_sub_callback_(const nav_msgs::msg::Odometry::SharedPtr msg) 
{
    eta_(0) = msg->pose.pose.position.x;
    eta_(1) = msg->pose.pose.position.y;
    eta_(2) = msg->pose.pose.position.z;
    Eigen::Quaterniond q(
        msg->pose.pose.orientation.w,
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z);
    Eigen::Vector3d euler = quaternion_to_euler(q);
    eta_(3) = euler(0);
    eta_(4) = euler(1);
    eta_(5) = euler(2);

    nu_(0) = msg->twist.twist.linear.x;
    nu_(1) = msg->twist.twist.linear.y;
    nu_(2) = msg->twist.twist.linear.z;
    nu_(3) = msg->twist.twist.angular.x;
    nu_(4) = msg->twist.twist.angular.y;
    nu_(5) = msg->twist.twist.angular.z;
}

void ControlMux::timer_callback_() 
{
    wrench_msg.header.stamp = this->now();
    wrench_noise_msg.header.stamp = this->now();

    // Calculate ocean current disturbance if enabled
    if (oc_config_.enable && oc_generator_) 
    {
        oc_disturbance = oc_generator_->step_tau(eta_, nu_, auv_dyn);
    }
    else
    {
        oc_disturbance.setZero();
    }

    switch (control_mode_) 
    {
        case ControlMode::OPEN_LOOP:
            if (is_signal_gen_active_) 
            {
                wrench_noise_ = ext_signal_.row(signal_index_).transpose();
                wrench_cmd_ = wrench_noise_;
                signal_index_++;
            } 
            else 
            {
                wrench_noise_.setZero();
                wrench_cmd_.setZero();
            }
            break;

        case ControlMode::CLOSED_LOOP:
            wrench_cmd_ = wrench_desired_;
            if (is_signal_gen_active_) 
            {
                wrench_noise_ = ext_signal_.row(signal_index_).transpose();
                wrench_cmd_ += wrench_noise_;
                signal_index_++;
            }
            else 
            {
                wrench_cmd_ = wrench_desired_;
                wrench_noise_.setZero();
            }
            break;

        default:
            RCLCPP_ERROR(this->get_logger(), "Unknown control mode!");
            wrench_cmd_.setZero();
            wrench_noise_.setZero();
            break;
    }

    // Add offset to last commanded torques if enabled
    if (enable_offset_) {
        wrench_cmd_ += wrench_offset_;
    }

    // Add ocean current disturbance if enabled
    if (oc_config_.enable) {
        wrench_cmd_ += oc_disturbance;
    }

    // Noise wrench message
    wrench_noise_msg.wrench.force.x = wrench_noise_(0);
    wrench_noise_msg.wrench.force.y = wrench_noise_(1);
    wrench_noise_msg.wrench.force.z = wrench_noise_(2);
    wrench_noise_msg.wrench.torque.x = wrench_noise_(3);
    wrench_noise_msg.wrench.torque.y = wrench_noise_(4);
    wrench_noise_msg.wrench.torque.z = wrench_noise_(5);

    // Ocean current disturbance message
    ocean_current_msg.wrench.force.x = oc_disturbance(0);
    ocean_current_msg.wrench.force.y = oc_disturbance(1);
    ocean_current_msg.wrench.force.z = oc_disturbance(2);
    ocean_current_msg.wrench.torque.x = oc_disturbance(3);
    ocean_current_msg.wrench.torque.y = oc_disturbance(4);
    ocean_current_msg.wrench.torque.z = oc_disturbance(5);
    
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
    ocean_current_pub_->publish(ocean_current_msg);

    // Check if signal generation is completed
    if (is_signal_gen_active_ && signal_index_ >= ext_signal_.rows()) 
    {
        is_signal_gen_active_ = false;
        signal_index_ = 0;
        RCLCPP_INFO(this->get_logger(), "Signal generation completed.");

        // Stop data recording via client call
        if (record_client_ -> service_is_ready()) 
        {
            RCLCPP_INFO(this->get_logger(), "Stopping data recording via record service.");
            record_request_->data = false; 
            auto result_future = record_client_->async_send_request(record_request_);
        } 
        else 
        {
            RCLCPP_WARN(this->get_logger(), "Record service is not available to stop recording.");
        }
    }
}

std::tuple<bool, Eigen::MatrixXd> ControlMux::generate_signal_(int n_samples, int n_signals, SignalGenerator::SignalType signal_type) 
{
    // Declare output variables
    bool is_gen = false;
    Eigen::MatrixXd ext_signal = {};

    switch (signal_type)
    {
        case SignalGenerator::SignalType::RBS:
            is_gen = true;
            ext_signal = SignalGenerator::RBS(n_samples, n_signals_, rbs_config_);
            RCLCPP_INFO(this->get_logger(), "############### Generated RBS signal ###############");
            RCLCPP_INFO_STREAM(this->get_logger(), "- max: " << rbs_config_.max.transpose());
            RCLCPP_INFO_STREAM(this->get_logger(), "- min: " << rbs_config_.min.transpose());
            break;

        case SignalGenerator::SignalType::RGS:
            is_gen = true;
            ext_signal = SignalGenerator::RGS(n_samples, n_signals_, rgs_config_);
            RCLCPP_INFO(this->get_logger(), "############### Generated RGS signal ###############");
            RCLCPP_INFO_STREAM(this->get_logger(), "- mean: " << rgs_config_.mean.transpose());
            RCLCPP_INFO_STREAM(this->get_logger(), "- stddev: " << rgs_config_.stddev.transpose());
            break;

        case SignalGenerator::SignalType::MULTISINE:
            is_gen = true;
            ext_signal = SignalGenerator::Multisine(n_samples, n_signals_, multisine_config_);
            RCLCPP_INFO(this->get_logger(), "############### Generated Multisine signal ###############");
            RCLCPP_INFO(this->get_logger(), "- n_sines: %d", multisine_config_.n_sines);
            RCLCPP_INFO(this->get_logger(), "- grid_skips: %d", multisine_config_.grid_skips);
            RCLCPP_INFO(this->get_logger(), "- n_trails: %d", multisine_config_.n_trails);
            RCLCPP_INFO_STREAM(this->get_logger(), "- amplitudes: " << multisine_config_.amplitudes.transpose());
            break;

        case SignalGenerator::SignalType::CHIRP:
            is_gen = true;
            ext_signal = SignalGenerator::Chirp(n_samples, n_signals_, chirp_config_);
            RCLCPP_INFO(this->get_logger(), "Generated Chirp signal");
            break;

        case SignalGenerator::SignalType::PRBS:
            is_gen = true;
            ext_signal = SignalGenerator::PRBS(n_samples, n_signals_, prbs_config_);
            RCLCPP_INFO(this->get_logger(), "Generated PRBS signal");
            break;

        default:
            is_gen = false;
            ext_signal = Eigen::MatrixXd::Zero(0, 0);
            RCLCPP_ERROR(this->get_logger(), "Unknown signal type!");
            break;
    }
    return {is_gen, ext_signal};
}

void ControlMux::signal_gen_trigger_srv_callback_(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response) 
{
    RCLCPP_INFO(this->get_logger(), "Signal generation triggered via service call.");
    int n_samples = static_cast<int>(duration_ / dt_);
    auto [is_gen, ext_signal] = generate_signal_(n_samples, n_signals_, signal_type_);

    if (is_gen) 
    {
        is_signal_gen_active_ = true;
        ext_signal_ = ext_signal;
        signal_index_ = 0;
        response->success = true;
        response->message = "Signal generation started.";
    } 
    else 
    {
        is_signal_gen_active_ = false;
        response->success = false;
        response->message = "Signal generation failed due to unknown signal type.";
    }

    // Check client is available and start data recording
    if (record_client_ -> service_is_ready())
    {
        if (signal_index_ == 0) 
        {
            RCLCPP_INFO(this->get_logger(), "Triggering data recording via record_trigger service.");
            record_request_->data = true; 
            auto result_future = record_client_->async_send_request(record_request_);
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "Signal generation is already active. Data recording not triggered.");
        }
    }
    else 
    {
        RCLCPP_WARN(this->get_logger(), "Record service is not available to start recording.");
    }
}

void ControlMux::inject_control_noise_srv_callback_(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response) 
{
    if (request->data)
    {
        RCLCPP_INFO(this->get_logger(), "Inject control noise service called. Setting signal generation active.");
        int n_samples = static_cast<int>(duration_ / dt_);
        auto [is_gen, ext_signal] = generate_signal_(n_samples, n_signals_, signal_type_);

        if (is_gen) 
        {
            is_signal_gen_active_ = true;
            ext_signal_ = ext_signal;
            signal_index_ = 0;
            response->success = true;
            response->message = "Signal generation started.";
        } 
        else 
        {
            is_signal_gen_active_ = false;
            response->success = false;
            response->message = "Signal generation failed due to unknown signal type.";
        }
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "Inject control noise service called. Stopping signal generation.");
        is_signal_gen_active_ = false;
        response->success = true;
        response->message = "Signal generation stopped.";
    }
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