#include "kernels/base.hpp"
#include <functional>
#include "logging/logging.hpp"

#ifndef ADAPTIVE_RWM_KERNEL_H
class AdaptiveRWMKernel : public TransitionKernel {
public:
	AdaptiveRWMKernel(
		std::function<double(const Eigen::VectorXd &)> log_likelihood, 
		std::function<double(const Eigen::VectorXd &)> log_prior,
		const Eigen::VectorXd sigma
		);
public:
	virtual std::unique_ptr<OneKernelResult> apply(const Eigen::VectorXd &sample, const int n_transitions, const bool return_path=true);
	virtual double log_target(const Eigen::VectorXd &sample) const;
	virtual void adapt();
private:
	std::function<double(const Eigen::VectorXd &)> log_likelihood;
	std::function<double(const Eigen::VectorXd &)> log_prior;
	const Eigen::VectorXd sigma;
	double _sigma_factor = 1.0;
	double _cur_accept_rate = 0.0;
	int _cur_transitions = 0;
};

#define ADAPTIVE_RWM_KERNEL_H
#endif