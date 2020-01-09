#include "kernels/base.hpp"
#include <functional>
#include <string>

#ifndef RWM_KERNEL_H
class RWMKernel : public TransitionKernel {
public:
	RWMKernel(
		std::function<double(const Eigen::VectorXd &)> log_likelihood, 
		std::function<double(const Eigen::VectorXd &)> log_prior,
		const Eigen::VectorXd sigma
		);
public:
	virtual std::unique_ptr<OneKernelResult> apply(const Eigen::VectorXd &sample, const int n_transitions, const bool return_path=true);
	virtual double log_target(const Eigen::VectorXd &sample) const;
private:
	std::function<double(const Eigen::VectorXd &)> log_likelihood;
	std::function<double(const Eigen::VectorXd &)> log_prior;
	const Eigen::VectorXd sigma;
};

#define RWM_KERNEL_H
#endif