#ifndef __SIGNAL_GEN_LIB_HPP__
#define __SIGNAL_GEN_LIB_HPP__

#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Core>

namespace SysID 
{
    namespace SignalGenerator
    {
        enum class SignalType 
        {
            RBS,
            RGS,
            CHIRP,
            PRBS,
            MULTISINE,
            UNKNOWN
        };

        // Helper function to convert string to enum
        inline SignalType stringToSignalType(const std::string& str) 
        {
            if (str == "RBS") return SignalType::RBS;
            if (str == "RGS") return SignalType::RGS;
            if (str == "Chirp") return SignalType::CHIRP;
            if (str == "PRBS") return SignalType::PRBS;
            if (str == "Multisine") return SignalType::MULTISINE;
            return SignalType::UNKNOWN;
        }

        struct RBSConfig
        {
            Eigen::Vector<double, 6> max = Eigen::VectorXd::Zero(6);
            Eigen::Vector<double, 6> min = Eigen::VectorXd::Zero(6);
        };

        struct RGSConfig
        {
            Eigen::Vector<double, 6> mean = Eigen::VectorXd::Zero(6);
            Eigen::Vector<double, 6> stddev = Eigen::VectorXd::Zero(6);
        };

        struct MultisineConfig
        {
            int n_sines = 0;            // Number of sinusoids
            int grid_skips = 0;         // Number of frequency grid skips
            int n_trails = 0;           // Number of trails 
            Eigen::Vector<double, 6> amplitudes = Eigen::VectorXd::Zero(6);
        };

        struct ChirpConfig
        {
        };

        struct PRBSConfig
        {
        };

        Eigen::MatrixXd RBS(const int &n_samples, const int &n_signals, const RBSConfig &config);
        Eigen::MatrixXd RGS(const int &n_samples, const int &n_signals, const RGSConfig &config);
        Eigen::MatrixXd Multisine(const int &n_samples, const int &n_signals, const MultisineConfig &config);
        Eigen::MatrixXd Chirp(const int &n_samples, const int &n_signals, const ChirpConfig &config);
        Eigen::MatrixXd PRBS(const int &n_samples, const int &n_signals, const PRBSConfig &config);

    };
}
#endif // __SIGNAL_GEN_LIB_HPP__
