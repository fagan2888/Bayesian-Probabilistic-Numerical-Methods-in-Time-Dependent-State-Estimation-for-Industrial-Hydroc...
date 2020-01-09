#include "kernels/base.hpp"
#include <memory>

#ifndef PCN_KERNEL_H

class PCNKernel : public TransitionKernel {
public:
	PCNKernel(
		double beta,
		std::function<double(const Eigen::VectorXd &)> log_likelihood_function, 
		const Eigen::VectorXd prior_mean,
		const Eigen::MatrixXd sqrt_prior_cov
	);
	virtual std::unique_ptr<OneKernelResult> apply(const Eigen::VectorXd &sample, const int n_transitions, const bool return_path=true);
	virtual double log_target(const Eigen::VectorXd &sample) const;
private:
	double _beta;
	std::function<double(const Eigen::VectorXd &)> _log_likelihood_function;
	const Eigen::VectorXd _prior_mean;
	const Eigen::MatrixXd _sqrt_prior_cov;
};

#define PCN_KERNEL_H
#endif