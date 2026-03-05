#include "xplorer_mini_sysid/ocean_current_lib.hpp"

namespace OceanEnvironment 
{
    OceanCurrentGenerator::OceanCurrentGenerator(double dt, const OceanCurrentConfig& config, int seed)
        : dt_(dt), config_(config), Vc_(config.Vc.init_val), alpha_(config.alpha.init_val), beta_(config.beta.init_val) 
    {
        if (seed >= 0) {
            generator_.seed(seed);
        } else {
            std::random_device rd;
            generator_.seed(rd());
        }
    }

    double OceanCurrentGenerator::step_variable_(double current_val, const OCParamConfig& conf) 
    {
        double stddev = std::sqrt(conf.var_w / dt_);
        std::normal_distribution<double> dist(conf.mean_w, stddev);
        double w = dist(generator_);
        
        double x_dot = -conf.mu * current_val + w;
        double next_val = current_val + x_dot * dt_;
        
        if (next_val > conf.max_val) next_val = conf.max_val;
        if (next_val < conf.min_val) next_val = conf.min_val;
        
        return next_val;
    }

    Eigen::Matrix<double, 6, 1> OceanCurrentGenerator::step_nu() 
    {
        Vc_ = step_variable_(Vc_, config_.Vc);
        alpha_ = step_variable_(alpha_, config_.alpha);
        beta_ = step_variable_(beta_, config_.beta);

        Eigen::Matrix<double, 6, 1> nu_c = Eigen::VectorXd::Zero(6);
        nu_c(0) = Vc_ * std::cos(alpha_) * std::cos(beta_);
        nu_c(1) = Vc_ * std::sin(beta_);
        nu_c(2) = Vc_ * std::sin(alpha_) * std::cos(beta_);
        
        return nu_c;
    }

    Eigen::Matrix<double, 6, 1> OceanCurrentGenerator::step_tau(const Eigen::Matrix<double, 6, 1>& eta, const Eigen::Matrix<double, 6, 1>& nu, const AUVDynamicParams& auv_params)
    {
        // ---- Calculate Ocean current simulation (NED) ----
        Eigen::Matrix<double, 6, 1> nu_c_sim = step_nu();
        
        // ----Calculate relative velocity simulation (BODY-FRAME IN NED) ----
        Eigen::Matrix<double, 6, 1> nu_r_sim = vector6_to_ned(nu) - (eulerang(eta.tail(3)).inverse() * nu_c_sim);

        // ---- Calculate Dynamic ----
        // Coriolis Matrix
        Eigen::Matrix<double, 6, 6> c_rb_nu = m2c(auv_params.m_rb_mat, nu);
        Eigen::Matrix<double, 6, 6> c_a_nu = m2c(auv_params.m_a_mat, nu);
        Eigen::Matrix<double, 6, 6> c_rb_nu_r = m2c(auv_params.m_rb_mat, nu_r_sim);
        Eigen::Matrix<double, 6, 6> c_a_nu_r = m2c(auv_params.m_a_mat, nu_r_sim);

        // Damping Matrix
        Eigen::Matrix<double, 6, 6> d_nu = auv_params.d_l_mat + (auv_params.d_nl_mat * nu.cwiseAbs().asDiagonal());
        Eigen::Matrix<double, 6, 6> d_nu_r = auv_params.d_l_mat + (auv_params.d_nl_mat * nu_r_sim.cwiseAbs().asDiagonal());

        // ---- Calculate current-induced forces and torques based on ocean current ----
        Eigen::Matrix<double, 6, 1> tau_c_ned = Eigen::VectorXd::Zero(6);
        Eigen::Matrix<double, 6, 1> tau_c_enu = Eigen::VectorXd::Zero(6);
        tau_c_ned = ((c_rb_nu + c_a_nu) * nu - (c_rb_nu_r + c_a_nu_r) * nu_r_sim) + ((d_nu * nu) - (d_nu_r * nu_r_sim));
        tau_c_enu = vector6_to_ned(tau_c_ned);
        
        return tau_c_enu;
    }
}