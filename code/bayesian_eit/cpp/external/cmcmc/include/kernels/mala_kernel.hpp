#include "kernels/base.hpp"
#include <memory>

#ifndef MALA_KERNEL_H

class MalaKernel : public TransitionKernel {
public:
	MalaKernel(
		std::function<GradientEval(const Eigen::VectorXd &)> log_likelihood, 
		std::function<GradientEval(const Eigen::VectorXd &)> log_prior,
		const Eigen::VectorXd preconditioner,
		const double tau
	);
	virtual std::unique_ptr<OneKernelResult> apply(const Eigen::VectorXd &sample, const int n_transitions, const bool return_path=true);
	virtual double log_target(const Eigen::VectorXd &sample) const;
private:
	std::function<GradientEval(const Eigen::VectorXd &)> log_likelihood;
	std::function<GradientEval(const Eigen::VectorXd &)> log_prior;
	const Eigen::VectorXd preconditioner;
	const double tau;
};

#define MALA_KERNEL_H
#endif