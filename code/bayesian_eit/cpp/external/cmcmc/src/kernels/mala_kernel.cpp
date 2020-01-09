#include "kernels/mala_kernel.hpp"
#include "mala.hpp"
#include "kernel_result.hpp"
#include <memory>


MalaKernel::MalaKernel(
		std::function<GradientEval(const Eigen::VectorXd &)> log_likelihood, 
		std::function<GradientEval(const Eigen::VectorXd &)> log_prior,
		const Eigen::VectorXd preconditioner,
		const double tau
		) : log_likelihood(log_likelihood), log_prior(log_prior), preconditioner(preconditioner), tau(tau)
{ }

std::unique_ptr<OneKernelResult> MalaKernel::apply(const Eigen::VectorXd &sample, const int n_transitions, const bool return_path) 
{
	return apply_one_kernel_mala(sample, n_transitions, tau, log_likelihood, log_prior, preconditioner, return_path);
} 


double MalaKernel::log_target(const Eigen::VectorXd &sample) const
{
	auto likelihood = log_likelihood(sample);
	auto prior = log_prior(sample);
	return likelihood.value + prior.value;
}
