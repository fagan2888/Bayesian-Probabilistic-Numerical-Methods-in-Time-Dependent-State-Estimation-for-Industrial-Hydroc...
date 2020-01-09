#include "kernels/adaptive_rwm_kernel.hpp"
#include "rwm.hpp"
#include "kernel_result.hpp"
#include <iostream>

AdaptiveRWMKernel::AdaptiveRWMKernel(
		std::function<double(const Eigen::VectorXd &)> log_likelihood, 
		std::function<double(const Eigen::VectorXd &)> log_prior,
		const Eigen::VectorXd sigma
		) : log_likelihood(log_likelihood), log_prior(log_prior), sigma(sigma)
{ }

std::unique_ptr<OneKernelResult> AdaptiveRWMKernel::apply(const Eigen::VectorXd &sample, const int n_transitions, const bool return_path) 
{
	auto ret = apply_one_kernel_rwm(sample, n_transitions, _sigma_factor*sigma, log_likelihood, log_prior, return_path);
	
	_cur_accept_rate = (_cur_transitions*_cur_accept_rate + n_transitions*ret->average_acceptance) / (_cur_transitions + n_transitions);
	_cur_transitions += n_transitions;

	return ret;
}

void AdaptiveRWMKernel::adapt() {
	if(_cur_accept_rate < 0.1) {
		_sigma_factor /= 2;
		LOG_INFO(label << " adapted to " << _sigma_factor << " due to accept rate " << _cur_accept_rate);
	} 
	if(_cur_accept_rate > 0.5) {
		_sigma_factor *= 2;
		LOG_INFO(label << " adapted to " << _sigma_factor << " due to accept rate " << _cur_accept_rate);
	}
	/*
	else
		LOG_INFO(label << " needed no adaptation, accept was " << _cur_accept_rate);
	*/
	_cur_transitions = 0;
	_cur_accept_rate = 0.0;
}

double AdaptiveRWMKernel::log_target(const Eigen::VectorXd &sample) const
{
	return log_likelihood(sample) + log_prior(sample);
}