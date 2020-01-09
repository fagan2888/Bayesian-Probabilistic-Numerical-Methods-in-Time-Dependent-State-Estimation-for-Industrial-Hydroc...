#include "kernels/pcn_kernel.hpp"
#include "pcn.hpp"
#include "kernel_result.hpp"
#include <memory>

PCNKernel::PCNKernel(
		double beta,
		std::function<double(const Eigen::VectorXd &)> log_likelihood_function, 
		const Eigen::VectorXd prior_mean,
		const Eigen::MatrixXd sqrt_prior_cov
	) : _beta(beta), _log_likelihood_function(log_likelihood_function), _prior_mean(prior_mean), _sqrt_prior_cov(sqrt_prior_cov)
{ }

std::unique_ptr<OneKernelResult> PCNKernel::apply(const Eigen::VectorXd &sample, const int n_transitions, const bool return_path) {
	return apply_one_kernel_pcn(sample, n_transitions, _beta, _log_likelihood_function, _prior_mean, _sqrt_prior_cov, return_path);
}

double PCNKernel::log_target(const Eigen::VectorXd &sample) const {
	return _log_likelihood_function(sample);
}