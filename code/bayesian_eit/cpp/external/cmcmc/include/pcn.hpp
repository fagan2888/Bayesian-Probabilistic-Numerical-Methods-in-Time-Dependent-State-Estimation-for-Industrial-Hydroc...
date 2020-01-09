#include <Eigen/Core>
#include <functional>
#include <memory>
#include "kernel_result.hpp"

#ifndef PCN_H

std::unique_ptr<OneKernelResult> apply_one_kernel_pcn(
	const Eigen::VectorXd &sample,
	int n_transitions,
	double beta,
	std::function<double(Eigen::VectorXd &)> log_likelihood_function,
	const Eigen::MatrixXd &sqrt_prior_cov,
	bool return_samples
);
std::unique_ptr<OneKernelResult> apply_one_kernel_pcn(
	const Eigen::VectorXd &sample,
	int n_transitions,
	double beta,
	std::function<double(Eigen::VectorXd &)> log_likelihood_function,
	const Eigen::VectorXd &prior_mean,
	const Eigen::MatrixXd &sqrt_prior_cov,
	bool return_samples
);

#define PCN_H
#endif