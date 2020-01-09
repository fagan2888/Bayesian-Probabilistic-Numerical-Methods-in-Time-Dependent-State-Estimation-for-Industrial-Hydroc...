#include <functional>
#include "kernels/base.hpp"
#include "parallel_tempering.hpp"
#include "mala.hpp"
#include "memory_utils.hpp"
#include "debug.hpp"
#include <iostream>
#include <vector>
#include <random>

std::unique_ptr<ParallelTemperingResult> apply_kernel_pt(
	const Eigen::MatrixXd &x0, 
	const int n_transitions,
	const std::vector<std::unique_ptr<TransitionKernel>> &kernels,
	int n_threads
) 
{
	Eigen::MatrixXd terminal_samples(x0.rows(), x0.cols());
	Eigen::MatrixXd target_path;
	Eigen::VectorXd terminal_log_targets(x0.rows());
	Eigen::VectorXd acceptances(x0.rows());

	#pragma omp parallel for num_threads(n_threads)
	for(int i = 0; i < x0.rows(); i++) {
		// I don't want ownership of the kernel so just get the raw pointer
		TransitionKernel *kernel = kernels.at(i).get();
		LOG_DEBUG("Applying kernel " << kernel->label);
		auto sample = x0.row(i);

		auto result = kernel->apply(sample, n_transitions, false);
		terminal_samples.row(i) = result->result;
		terminal_log_targets(i) = result->log_target;
		acceptances(i) = result->average_acceptance;
	}
	auto ret = make_unique<ParallelTemperingResult>(terminal_samples, target_path, terminal_log_targets, acceptances);
	return ret;
}

std::vector<std::unique_ptr<Eigen::MatrixXd>> apply_pt(
	const Eigen::MatrixXd &x0,
	const int transitions_per_swap,
	const int n_swaps,
	const std::vector<int> target_ixs,
	const std::vector<std::unique_ptr<TransitionKernel>> &kernels,
	int n_threads,
	int thinning
)
{
	const int adapt_every = 100;

	std::default_random_engine generator(std::random_device{}());
	std::uniform_real_distribution<double> uniform_distribution(0.0,1.0);
	// range is inclusive
	std::uniform_int_distribution<int> index_distribution(0, x0.rows() - 2);

	Eigen::MatrixXd cur_sample = x0;
	int n_out = (int)(n_swaps * 1./thinning);
	std::vector<std::unique_ptr<Eigen::MatrixXd>> ret;
	for(int i = 0; i < target_ixs.size(); i++) {
		std::unique_ptr<Eigen::MatrixXd> ret_matrix = make_unique<Eigen::MatrixXd>(n_out, x0.cols());
		ret.push_back(std::move(ret_matrix));
	}
	for(int i = 0; i < n_swaps; i++) {
		// apply the kernel
		auto result = apply_kernel_pt(cur_sample, transitions_per_swap, kernels, n_threads);
		cur_sample = result->end_samples;
		// TODO: randomly select an index to swap
		int i1 = index_distribution(generator);
		// compute the hastings ratio
		int i2 = i1 + 1;

		auto s1 = cur_sample.row(i1);
		auto s2 = cur_sample.row(i2);

		double log_target_11 = result->terminal_log_targets(i1);
		double log_target_22 = result->terminal_log_targets(i2);

		double log_target_12 = kernels.at(i1).get()->log_target(s2);
		double log_target_21 = kernels.at(i2).get()->log_target(s1);

		double log_accept_ratio = log_target_12 + log_target_21 - log_target_11 - log_target_22;

		bool accept = uniform_distribution(generator) < exp(log_accept_ratio);

		if(accept) {
			cur_sample.row(i2) = s1;
			cur_sample.row(i1) = s2;
		}
		//std::cout << i << ": Proposed swap between " << i1 << " and " << i2 << " was " << (accept ? "" : "not ") << "accepted." << std::endl;
		//std::cout << result->acceptances << std::endl;

		if (i % thinning == 0)
		{
			int out_row = (int)(i * 1./thinning);
			for(int j = 0; j < target_ixs.size(); j++){
				int target_ix = target_ixs.at(j);
				// not taking ownership
				Eigen::MatrixXd *ret_matrix = ret.at(j).get();
				ret_matrix->row(out_row) = cur_sample.row(target_ix);
			}
		}
		if (i % adapt_every == 0)
		{
			for(int j = 0; j < x0.rows(); j++)
				kernels.at(j).get()->adapt();
		}
		if (i % (n_swaps / 100) == 0) {
			LOG_INFO((int)(i / (float) n_swaps * 100)  << "\% complete.");
		}
	}

	return ret;
}