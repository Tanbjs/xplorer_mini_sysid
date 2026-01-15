#ifndef __SIGNAL_GEN_LIB_HPP__
#define __SIGNAL_GEN_LIB_HPP__

#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Core>

namespace SysID 
{
    namespace SignalGen 
    {
        Eigen::MatrixXd PRBS(
            const int &n_samples,
            const int &n_signals,
            const int &min_switch_time,
            const int &max_switch_time,
            const Eigen::VectorXd &amplitudes
        );

        Eigen::MatrixXd UniformWhiteNoise(
            const int &n_samples,
            const int &n_signals,
            const Eigen::VectorXd &amplitudes
        );

        Eigen::MatrixXd Multisine(
            const int &n_samples,
            const int &n_signals,
            const int &f_max,
            const Eigen::VectorXd &amplitudes
        );

        Eigen::MatrixXd Chirp(
            const int &n_samples,
            const int &n_signals,
            const double &f_start,
            const double &f_end,
            const Eigen::VectorXd &amplitudes
        );
    };
}
#endif // __SIGNAL_GEN_LIB_HPP__
