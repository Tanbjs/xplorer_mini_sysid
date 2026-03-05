#ifndef __OCEAN_CURRENT_LIB_HPP__
#define __OCEAN_CURRENT_LIB_HPP__

#include <cmath>
#include <random>

#include "xplorer_mini_cpp_utils/kinematics_utils.hpp"

namespace OceanEnvironment 
{
    struct OCParamConfig 
    {
        double mu;
        double mean_w;
        double var_w;
        double init_val;
        double min_val;
        double max_val;
    };

    struct OceanCurrentConfig 
    {
        bool enable;
        int seed;
        OCParamConfig Vc;
        OCParamConfig alpha;
        OCParamConfig beta;
    };

    class OceanCurrentGenerator 
    {
    public:
        OceanCurrentGenerator(double dt, const OceanCurrentConfig& config, int seed);
        Eigen::Matrix<double, 6, 1> step_nu();
        Eigen::Matrix<double, 6, 1> step_tau(const Eigen::Matrix<double, 6, 1>& eta, const Eigen::Matrix<double, 6, 1>& nu, const AUVDynamicParams& auv_params);

    private:
        double dt_;
        OceanCurrentConfig config_;
        double Vc_, alpha_, beta_;
        std::default_random_engine generator_;
        
        double step_variable_(double current_val, const OCParamConfig& conf);
    };
};

#endif // __OCEAN_CURRENT_LIB_HPP__