#ifndef SIMULATE_H
#include <vector>
#include <memory>

struct SimulateResult {
	const Eigen::MatrixXd samples;
	const Eigen::VectorXd acceptances;
	const Eigen::VectorXd log_likelihoods;
	std::vector<std::unique_ptr<Eigen::MatrixXd>> sample_paths;

	SimulateResult(const Eigen::MatrixXd samples, const Eigen::VectorXd acceptances, const Eigen::VectorXd log_likelihoods, 
		std::vector<std::unique_ptr<Eigen::MatrixXd>> sample_paths) 
		: samples(samples), acceptances(acceptances), log_likelihoods(log_likelihoods), sample_paths(std::move(sample_paths)) { }
};

std::unique_ptr<SimulateResult> run_pcn_parallel(
	int n_iter,
	double beta,
	const Eigen::Ref<const Eigen::MatrixXd> &theta_0,
	const Eigen::Ref<const Eigen::VectorXd> &prior_mean,
	const Eigen::Ref<const Eigen::MatrixXd> &sqrt_prior_cov,
	const Eigen::Ref<const Eigen::MatrixXd> &interior,
	const Eigen::Ref<const Eigen::MatrixXd> &sensors,
	const Eigen::Ref<const Eigen::MatrixXd> &theta_projection_mat,
	const Eigen::Ref<const Eigen::VectorXd> &kernel_args,
	const Eigen::Ref<const Eigen::MatrixXd> &stim_pattern,
	const Eigen::Ref<const Eigen::MatrixXd> &meas_pattern,
	const Eigen::Ref<const Eigen::MatrixXd> &data,
	double likelihood_variance,
	int n_threads,
	bool return_samples,
	bool bayesian=true
);

std::unique_ptr<SimulateResult> run_pcn_parallel_tempered(
	int n_iter,
	double beta,
	const Eigen::Ref<const Eigen::MatrixXd> &theta_0,
	const Eigen::Ref<const Eigen::VectorXd> &prior_mean,
	const Eigen::Ref<const Eigen::MatrixXd> &sqrt_prior_cov,
	const Eigen::Ref<const Eigen::MatrixXd> &interior,
	const Eigen::Ref<const Eigen::MatrixXd> &sensors,
	const Eigen::Ref<const Eigen::MatrixXd> &theta_projection_mat,
	const Eigen::Ref<const Eigen::VectorXd> &kernel_args,
	const Eigen::Ref<const Eigen::MatrixXd> &stim_pattern,
	const Eigen::Ref<const Eigen::MatrixXd> &meas_pattern,
	const Eigen::Ref<const Eigen::MatrixXd> &data_1,
	const Eigen::Ref<const Eigen::MatrixXd> &data_2,
	double temperature,
	double likelihood_variance,
	int n_threads,
	bool return_samples,
	bool bayesian=true
);

#endif