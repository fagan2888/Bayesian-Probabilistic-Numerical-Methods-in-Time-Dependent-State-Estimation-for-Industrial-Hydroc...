#include <memory>
#include <Eigen/Core>
#include "kernel_result.hpp"
#include <string>

#ifndef TRANSITION_KERNEL_H
class TransitionKernel {
public:
	virtual std::unique_ptr<OneKernelResult> apply(const Eigen::VectorXd &sample, const int n_transitions, const bool return_path=true) = 0;
	virtual double log_target(const Eigen::VectorXd &sample) const = 0;
	virtual void adapt() { }
	std::string label = "";
private:
	
};

#define TRANSITION_KERNEL_H
#endif