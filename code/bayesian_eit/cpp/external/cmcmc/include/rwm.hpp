#include <Eigen/Core>
#include <functional>
#include <memory>

#include "kernel_result.hpp"

#ifndef KERNELS_H

std::unique_ptr<KernelResult> apply_kernel_rwm(
	const Eigen::Ref<const Eigen::MatrixXd> &samples,
	const int n_transitions,
	const Eigen::VectorXd &sigma,
	std::function<double(Eigen::VectorXd &)> log_likelihood_function,
	std::function<double(Eigen::VectorXd &)> log_prior_function,
	const int n_threads
);

std::unique_ptr<OneKernelResult> apply_one_kernel_rwm(
	const Eigen::VectorXd &sample,
	int n_transitions,
	const Eigen::VectorXd &sigma,
	std::function<double(Eigen::VectorXd &)> log_likelihood_function,
	std::function<double(Eigen::VectorXd &)> log_prior_function,
	bool return_samples = true
);

#define KERNELS_H
#endif