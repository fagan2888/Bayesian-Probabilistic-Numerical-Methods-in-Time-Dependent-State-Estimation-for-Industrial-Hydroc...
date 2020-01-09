// TODO: just need this for GradientEval, nice to separate that into its own class
#include <kernel_result.hpp>
#ifndef PRIOR_H
#include <Eigen/Core>
enum class PriorType {GAUSSIAN, CAUCHY, UNIFORM};

double eval_log_prior(PriorType type, const Eigen::VectorXd &sample, const Eigen::VectorXd &loc, const Eigen::VectorXd &scale);
GradientEval eval_grad_log_prior(PriorType type, const Eigen::VectorXd &sample, const Eigen::VectorXd &loc, const Eigen::VectorXd &scale);


double log_gaussian_dist(const Eigen::VectorXd &sample, const Eigen::VectorXd &loc, const Eigen::VectorXd &scale);
Eigen::VectorXd grad_log_gaussian_dist(const Eigen::VectorXd &sample, const Eigen::VectorXd &loc, const Eigen::VectorXd &scale);
double log_cauchy_dist(const Eigen::VectorXd &sample, const Eigen::VectorXd &loc, const Eigen::VectorXd &scale);
Eigen::VectorXd grad_log_cauchy_dist(const Eigen::VectorXd &sample, const Eigen::VectorXd &loc, const Eigen::VectorXd &scale);
double log_uniform_dist(const Eigen::VectorXd &sample, const Eigen::VectorXd &loc, const Eigen::VectorXd &scale);
#define PRIOR_H
#endif