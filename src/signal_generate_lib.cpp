#include "xplorer_mini_sysid/signal_generate_lib.hpp"

namespace SysID 
{
    namespace SignalGenerator
    {
        Eigen::MatrixXd UniformWhiteNoise(
            const int &n_samples,
            const int &n_signals,
            const Eigen::VectorXd &amplitudes
        ) 
        {
            Eigen::MatrixXd signal = Eigen::MatrixXd::Zero(n_samples, n_signals);
            std::default_random_engine generator;
            for (int i = 0; i < n_signals; ++i) 
            {
                std::uniform_real_distribution<double> distribution(-amplitudes(i), amplitudes(i));
                for (int k = 0; k < n_samples; ++k) 
                {
                    signal(k, i) = distribution(generator);
                }
            }
            return signal;
        }
        
        Eigen::MatrixXd PRBS(
            const int &n_samples,
            const int &n_signals,
            const int &min_switch_time,
            const int &max_switch_time,
            const Eigen::VectorXd &amplitudes
        ) 
        {
            Eigen::MatrixXd signal = Eigen::MatrixXd::Zero(n_samples, n_signals);
            std::default_random_engine generator;
            std::uniform_int_distribution<int> distribution(min_switch_time, max_switch_time);
            std::uniform_int_distribution<int> level_distribution(0, 1);

            for (int i = 0; i < n_signals; ++i) 
            {
                int k = 0;
                while (k < n_samples) 
                {
                    int switch_time = distribution(generator);
                    int level = level_distribution(generator) ? 1 : -1;
                    for (int j = 0; j < switch_time && k < n_samples; ++j, ++k) 
                    {
                        signal(k, i) = level * amplitudes(i);
                    }
                }
            }
            return signal;
        }
     
        Eigen::MatrixXd Multisine(
            const int &n_samples,
            const int &n_signals,
            const Eigen::VectorXd &f_max,
            const Eigen::VectorXd &amplitudes
        ) 
        {
            int n_signals = amplitudes.size();
            Eigen::MatrixXd signal = Eigen::MatrixXd::Zero(n_samples, n_signals);
            for (int i = 0; i < n_signals; ++i) 
            {
                for (int k = 0; k < n_samples; ++k) 
                {
                        double time = static_cast<double>(k);
                        signal(k, i) = 0.0;
                        for (double f = 1.0; f <= f_max(i); f += 1.0) 
                        {
                            signal(k, i) += amplitudes(i) * sin(2.0 * M_PI * f * time / n_samples);
                        }
                }
            }
            return signal;
        }
    };
}