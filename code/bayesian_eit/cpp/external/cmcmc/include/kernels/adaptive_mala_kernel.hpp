#include "kernels/base.hpp"
#include <memory>

#ifndef ADAPTIVE_MALA_KERNEL_H

class AdaptiveMalaKernel : public TransitionKernel {
public:
	AdaptiveMalaKernel(
		std::function<GradientEval(const Eigen::VectorXd &)> log_likelihood, 
		std::function<GradientEval(const Eigen::VectorXd &)> log_prior,
		const Eigen::VectorXd preconditioner,
		double tau
	);
public:
	virtual std::unique_ptr<OneKernelResult> apply(const Eigen::VectorXd &sample, const int n_transitions, const bool return_path=true);
	virtual double log_target(const Eigen::VectorXd &sample) const;
	virtual void adapt();
private:
	std::function<GradientEval(const Eigen::VectorXd &)> log_likelihood;
	std::function<GradientEval(const Eigen::VectorXd &)> log_prior;
	const Eigen::VectorXd preconditioner;
	double tau;
	double _cur_accept_rate = 0.0;
	int _cur_transitions = 0;
};

#define ADAPTIVE_MALA_KERNEL_H
#endif