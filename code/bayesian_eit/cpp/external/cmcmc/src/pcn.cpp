#include "memory_utils.hpp"
#include <math.h>
#include "pcn.hpp"
#include <random>

std::unique_ptr<OneKernelResult> apply_one_kernel_pcn(
	const Eigen::VectorXd &sample,
	int n_transitions,
	double beta,
	std::function<double(Eigen::VectorXd &)> log_likelihood_function,
	const Eigen::MatrixXd &sqrt_prior_cov,
	bool return_samples
)
{
	Eigen::VectorXd prior_mean = Eigen::VectorXd::Zero(sample.rows());
	return apply_one_kernel_pcn(sample, n_transitions, beta, log_likelihood_function, prior_mean, sqrt_prior_cov, return_samples);
}

std::unique_ptr<OneKernelResult> apply_one_kernel_pcn(
	const Eigen::VectorXd &sample,
	int n_transitions,
	double beta,
	std::function<double(Eigen::VectorXd &)> log_likelihood_function,
	const Eigen::VectorXd &prior_mean,
	const Eigen::MatrixXd &sqrt_prior_cov,
	bool return_samples
)
{

	std::default_random_engine generator(std::random_device{}());
	std::uniform_real_distribution<double> uniform_distribution(0.0,1.0);
	std::normal_distribution<double> gaussian_distribution(0.0, 1.);

	Eigen::MatrixXd ret_samples;
	Eigen::VectorXi acceptances;
	if(return_samples) {
		ret_samples = Eigen::MatrixXd(n_transitions, sample.rows());
		acceptances = Eigen::VectorXi(n_transitions);
	}
	Eigen::VectorXd random_noise(sqrt_prior_cov.cols());
	Eigen::VectorXd cur = sample;
	double cur_log_likelihood = log_likelihood_function(cur);
	Eigen::MatrixXd scaled_cov = beta*sqrt_prior_cov;
	int accepts = 0;
	
	double sqrt_1_m_beta = sqrt(1-beta*beta);

	for(int i = 0; i < n_transitions; i++) {
		// propose
		for(int j = 0; j < sqrt_prior_cov.cols(); j++) {
			random_noise(j) = gaussian_distribution(generator);
		}
		Eigen::VectorXd next = prior_mean + sqrt_1_m_beta*(cur-prior_mean) + scaled_cov * random_noise;

		// accept / reject
		double next_log_likelihood = log_likelihood_function(next);
		double accept_ratio = exp(next_log_likelihood - cur_log_likelihood);
		bool accept = uniform_distribution(generator) < accept_ratio;

		if(accept) {
			cur = next;
			cur_log_likelihood = next_log_likelihood;
			++accepts;
		}

		if(return_samples) {
			ret_samples.row(i) = cur;
			acceptances(i) = accept ? 1 : 0;
		}


	}
	auto res = make_unique<OneKernelResult>(
		cur, 
		ret_samples, 
		acceptances, 
		cur_log_likelihood,
		cur_log_likelihood,
		accepts * 1.0/((double)n_transitions)
	);
	return res;
}
