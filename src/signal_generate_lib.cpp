#include "xplorer_mini_sysid/signal_generate_lib.hpp"

namespace SysID 
{
    namespace SignalGenerator
    {
        Eigen::MatrixXd RBS(const int &n_samples, const int &n_signals, const RBSConfig &config) 
        {
            Eigen::MatrixXd signal = Eigen::MatrixXd::Zero(n_samples, n_signals);
            std::default_random_engine generator;
            for (int i = 0; i < n_signals; ++i) 
            {
                std::uniform_real_distribution<double> distribution(config.min(i), config.max(i));
                for (int k = 0; k < n_samples; ++k) 
                {
                    signal(k, i) = distribution(generator);
                }
            }
            return signal;
        }

        Eigen::MatrixXd RGS(const int &n_samples, const int &n_signals, const RGSConfig &config) 
        {
            Eigen::MatrixXd signal = Eigen::MatrixXd::Zero(n_samples, n_signals);
            std::default_random_engine generator;
            for (int i = 0; i < n_signals; ++i) 
            {
                std::normal_distribution<double> distribution(config.mean(i), config.stddev(i));
                for (int k = 0; k < n_samples; ++k) 
                {
                    signal(k, i) = distribution(generator);
                }
            }
            return signal;
        }

        Eigen::MatrixXd Multisine(const int &n_samples, const int &n_signals, const MultisineConfig &config) 
        {
        }

        Eigen::MatrixXd Chirp(const int &n_samples, const int &n_signals, const ChirpConfig &config) 
        {
        }

        Eigen::MatrixXd PRBS(const int &n_samples, const int &n_signals, const PRBSConfig &config) 
        {
        }

    };
};