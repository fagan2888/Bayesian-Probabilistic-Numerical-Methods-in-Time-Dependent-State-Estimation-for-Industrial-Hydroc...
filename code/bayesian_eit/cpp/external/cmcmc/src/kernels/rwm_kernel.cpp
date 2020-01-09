#include "kernels/rwm_kernel.hpp"
#include "rwm.hpp"
#include "kernel_result.hpp"
#include <iostream>

RWMKernel::RWMKernel(
		std::function<double(const Eigen::VectorXd &)> log_likelihood, 
		std::function<double(const Eigen::VectorXd &)> log_prior,
		const Eigen::VectorXd sigma
		) : log_likelihood(log_likelihood), log_prior(log_prior), sigma(sigma)
{ }

std::unique_ptr<OneKernelResult> RWMKernel::apply(const Eigen::VectorXd &sample, const int n_transitions, const bool return_path) 
{
	return apply_one_kernel_rwm(sample, n_transitions, sigma, log_likelihood, log_prior, return_path);
}

double RWMKernel::log_target(const Eigen::VectorXd &sample) const
{
	return log_likelihood(sample) + log_prior(sample);
}