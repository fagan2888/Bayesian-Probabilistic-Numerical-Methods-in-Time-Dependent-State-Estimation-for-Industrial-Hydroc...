#include <Eigen/Core>
#include <vector>
#include <memory>
#include "kernel_result.hpp"
#include "kernels/base.hpp"

std::unique_ptr<ParallelTemperingResult> apply_kernel_pt(
	const Eigen::MatrixXd &x0, 
	const int n_transitions,
	const std::vector<std::unique_ptr<TransitionKernel>> &kernels,
	int n_threads
);

std::vector<std::unique_ptr<Eigen::MatrixXd>> apply_pt(
	const Eigen::MatrixXd &x0,
	const int transitions_per_swap,
	const int n_swaps,
	const std::vector<int> target_ixs,
	const std::vector<std::unique_ptr<TransitionKernel>> &kernels,
	int n_threads,
	int thinning=1
);