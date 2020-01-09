#include <Eigen/Core>
#include <functional>
#include <memory>
#include "kernel_result.hpp"

#ifndef MALA_H

std::unique_ptr<KernelResult> apply_kernel_mala(
	const Eigen::Ref<const Eigen::MatrixXd> &samples,
	const int n_transitions,
	const double tau,
	std::function<GradientEval(Eigen::VectorXd &)> log_likelihood_function,
	std::function<GradientEval(Eigen::VectorXd &)> log_prior_function,
	const Eigen::VectorXd &preconditioner,
	const int n_threads
);

std::unique_ptr<OneKernelResult> apply_one_kernel_mala(
	const Eigen::VectorXd &sample,
	int n_transitions,
	const double tau,
	std::function<GradientEval(Eigen::VectorXd &)> log_likelihood_function,
	std::function<GradientEval(Eigen::VectorXd &)> log_prior_function,
	const Eigen::VectorXd &preconditioner,
	bool return_samples = true
);

#define MALA_H
#endif