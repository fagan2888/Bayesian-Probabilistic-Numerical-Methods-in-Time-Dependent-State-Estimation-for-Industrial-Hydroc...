#include <Eigen/Core>
#ifndef KERNEL_RESULT_H
struct KernelResult {
    Eigen::MatrixXd results;
    Eigen::VectorXd log_likelihoods;
    Eigen::VectorXd log_targets;
    Eigen::VectorXd acceptances;

    KernelResult(Eigen::MatrixXd results, Eigen::VectorXd log_likelihoods, Eigen::VectorXd log_targets, Eigen::VectorXd acceptances)
        : results(results), log_likelihoods(log_likelihoods), log_targets(log_targets), acceptances(acceptances) { }
};


struct OneKernelResult {
    Eigen::VectorXd result;
    Eigen::MatrixXd samples;
    Eigen::VectorXi acceptances;
    const double log_likelihood;
    const double log_target;
    const double average_acceptance;

    OneKernelResult(Eigen::VectorXd result, Eigen::MatrixXd samples, Eigen::VectorXi acceptances, const double log_likelihood, const double log_target, const double average_acceptance)
        : result(result), samples(samples), acceptances(acceptances), log_likelihood(log_likelihood), log_target(log_target), average_acceptance(average_acceptance)
    {}
};

struct GradientEval {
    double value;
    Eigen::VectorXd gradient;

    GradientEval(double value, Eigen::VectorXd gradient) 
        : value(value), gradient(gradient)
    { }
};

struct ParallelTemperingResult {
    Eigen::MatrixXd end_samples;
    Eigen::MatrixXd target_path;
    Eigen::VectorXd terminal_log_targets;
    Eigen::VectorXd acceptances;

    ParallelTemperingResult(Eigen::MatrixXd end_samples, Eigen::MatrixXd target_path, Eigen::VectorXd terminal_log_targets, Eigen::VectorXd acceptances)
        : end_samples(end_samples), target_path(target_path), terminal_log_targets(terminal_log_targets), acceptances(acceptances)
    { }
};

#define KERNEL_RESULT_H
#endif