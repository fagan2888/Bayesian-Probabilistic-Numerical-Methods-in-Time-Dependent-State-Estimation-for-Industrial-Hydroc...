#include <functional>
#include "likelihood.hpp"
#include "collocate.hpp"
#include "pcn.hpp"
#include "simulate.hpp"
#include "memory_utils.hpp"
#include <iostream>
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
	bool bayesian
)
{
	return run_pcn_parallel_tempered(
		n_iter,
		beta,
		theta_0,
		prior_mean,
		sqrt_prior_cov,
		interior,
		sensors,
		theta_projection_mat,
		kernel_args,
		stim_pattern,
		meas_pattern,
		data,
		Eigen::MatrixXd(0, 0),
		0.,
		likelihood_variance,
		n_threads,
		return_samples,
		bayesian
	);
}

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
	bool bayesian
)
{
	Eigen::MatrixXd ret_samples(theta_0.rows(), theta_0.cols());
	Eigen::VectorXd ret_likelihoods(theta_0.rows());
	Eigen::VectorXd acceptances(theta_0.rows());
	std::vector<std::unique_ptr<Eigen::MatrixXd>> sample_paths;
	#pragma omp parallel for num_threads(n_threads)
	for(int i = 0; i < theta_0.rows(); i++) {

		Collocator *collocator = new Collocator(sensors, interior.rows() + sensors.rows(), kernel_args);

		auto log_likelihood_function = [
			interior, 
			sensors, 
			theta_projection_mat, 
			kernel_args, 
			stim_pattern, 
			meas_pattern, 
			data_1,
			data_2,
			temperature, 
			likelihood_variance,
			collocator,
			bayesian
		] (const Eigen::VectorXd &theta) -> double {
			return log_likelihood_tempered(
				interior, 
				sensors, 
				theta, 
				theta_projection_mat,
				kernel_args, 
				stim_pattern, 
				meas_pattern, 
				data_1, 
				data_2, 
				temperature, 
				likelihood_variance, 
				collocator,
				bayesian
			);
		};

		Eigen::VectorXd sample = theta_0.row(i);
		auto results = apply_one_kernel_pcn(sample, n_iter, beta, log_likelihood_function, prior_mean, sqrt_prior_cov, return_samples);

		ret_samples.row(i) = results->result;
		ret_likelihoods(i) = results->log_likelihood;
		acceptances(i) = results->average_acceptance;
		
		if(return_samples) {
			Eigen::MatrixXd samplesMat = results->samples;
			std::unique_ptr<Eigen::MatrixXd> samples = make_unique<Eigen::MatrixXd>(samplesMat);
			sample_paths.push_back(std::move(samples));
		}
		delete collocator;
	}

	return make_unique<SimulateResult>(ret_samples, acceptances, ret_likelihoods, std::move(sample_paths));
}