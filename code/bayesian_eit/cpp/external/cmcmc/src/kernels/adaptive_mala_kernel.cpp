#include "kernels/adaptive_mala_kernel.hpp"
#include "mala.hpp"
#include "kernel_result.hpp"
#include <memory>
#include <iostream>
#include "logging/logging.hpp"

AdaptiveMalaKernel::AdaptiveMalaKernel(
		std::function<GradientEval(const Eigen::VectorXd &)> log_likelihood, 
		std::function<GradientEval(const Eigen::VectorXd &)> log_prior,
		const Eigen::VectorXd preconditioner,
		double tau
		) : log_likelihood(log_likelihood), log_prior(log_prior), preconditioner(preconditioner), tau(tau)
{ }

std::unique_ptr<OneKernelResult> AdaptiveMalaKernel::apply(const Eigen::VectorXd &sample, const int n_transitions, const bool return_path) 
{
	auto ret = apply_one_kernel_mala(sample, n_transitions, tau, log_likelihood, log_prior, preconditioner, return_path);

	_cur_accept_rate = (_cur_transitions*_cur_accept_rate + n_transitions*ret->average_acceptance) / (_cur_transitions + n_transitions);
	_cur_transitions += n_transitions;

	return ret;
}

void AdaptiveMalaKernel::adapt() {
	const double ADAPT_FACTOR = 1.25;
	if(_cur_accept_rate < 0.1) {
		tau /= ADAPT_FACTOR;
		LOG_INFO(label << " adapted to " << tau << " due to accept rate " << _cur_accept_rate);
	} 
	else if(_cur_accept_rate > 0.4) {
		tau *= ADAPT_FACTOR;
		LOG_INFO(label << " adapted to " << tau << " due to accept rate " << _cur_accept_rate);
	}
	/*
	else
		LOG_INFO(label << " needed no adaptation, accept was " << _cur_accept_rate);
	*/
	_cur_transitions = 0;
	_cur_accept_rate = 0.0;
}


double AdaptiveMalaKernel::log_target(const Eigen::VectorXd &sample) const
{
	auto likelihood = log_likelihood(sample);
	auto prior = log_prior(sample);
	return likelihood.value + prior.value;
}
