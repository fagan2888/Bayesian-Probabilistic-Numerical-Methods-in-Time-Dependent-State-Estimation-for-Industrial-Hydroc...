#include <iostream>
#include <random>
#include <memory>
#include "rwm.hpp"
#include "debug.hpp"
#include <math.h>
#include "memory_utils.hpp"

OneKernelResult _apply_one_kernel_rwm(
	const Eigen::VectorXd &sample,
	int n_transitions,
	const Eigen::VectorXd &sigma,
	std::function<double(Eigen::VectorXd &)> log_likelihood_function,
	std::function<double(Eigen::VectorXd &)> log_prior_function,
	bool return_samples
)
{
	double NEGINF = -1*std::numeric_limits<double>::infinity();;
	DEBUG_FUNCTION_ENTER;
	LOG_TRACE("Sample has " << sample.rows() << " rows");

    double new_log_likelihood, new_log_prior, accept_ratio;
    bool accept;
	Eigen::VectorXd cur = sample;
	Eigen::VectorXd next(cur.rows());
	
	double cur_log_likelihood = log_likelihood_function(cur);
	double cur_log_prior = log_prior_function(cur);

	LOG_TRACE("Initial log-likelihood is " << cur_log_likelihood);

	std::default_random_engine generator(std::random_device{}());
	std::uniform_real_distribution<double> uniform_distribution(0.0,1.0);

	Eigen::MatrixXd ret_samples;
	Eigen::VectorXi acceptances;
	if(return_samples) {
		ret_samples = Eigen::MatrixXd(n_transitions, sample.rows());
		acceptances = Eigen::VectorXi(n_transitions);
	}

	int accepts = 0;
	LOG_TRACE("About to start " << n_transitions << " RWM transitions with sigma=" << sigma);
	for(int i = 0; i < n_transitions; i++) {
		// this is shit but for some reason can't pass RNGs about...
		for(int j = 0; j < sample.rows(); j++) {
			std::normal_distribution<double> distribution(0.0, sigma(j));
			next(j) = cur(j) + distribution(generator);
		}

		new_log_likelihood = log_likelihood_function(next);
		new_log_prior = log_prior_function(next);

		LOG_TRACE("Iteration " << i << ": proposal done; it was " << next << " and has log-likelihood " << new_log_likelihood << ".");
		accept = false;
		// new sample is not in the support of the prior, reject
		if(new_log_prior == NEGINF) {
			LOG_TRACE("Not in support of prior; continuing.");
		}
		else if(new_log_likelihood == NEGINF) {
			LOG_TRACE("Potential is negative infinity; continuing.");
		}
		else if(cur_log_likelihood == NEGINF) {
			accept = true;
			LOG_TRACE("Old potential was neginf but new is not; accepting.");
		}
		// both current and new sample are in the ball, reject with probability based on the prior ratio.
		else {
            accept_ratio = exp(new_log_prior - cur_log_prior + new_log_likelihood - cur_log_likelihood);
            accept = uniform_distribution(generator) < accept_ratio;
            LOG_TRACE((accept ? "Accepted" : "Rejected") << " with probability " << accept_ratio);
        }

        if(accept) {
            accepts += 1;
            for(int j = 0; j < sample.rows(); j++) {
                cur(j) = next(j);
            }
            cur_log_likelihood = new_log_likelihood;
            cur_log_prior = new_log_prior;
		}

		if(return_samples)
		{
			ret_samples.row(i) = cur;
			acceptances(i) = accept ? 1 : 0;
		}
	}
	LOG_TRACE(accepts << " acceptances.");
	auto res = OneKernelResult(
		cur, 
		ret_samples, 
		acceptances, 
		cur_log_likelihood, 
		cur_log_likelihood + cur_log_prior, 
		accepts * 1.0/((double)n_transitions)
	);
	LOG_TRACE("Result contains " << res.result);
	
	DEBUG_FUNCTION_EXIT;
	return res;
}

std::unique_ptr<OneKernelResult> apply_one_kernel_rwm(
	const Eigen::VectorXd &sample,
	int n_transitions,
	const Eigen::VectorXd &sigma,
	std::function<double(Eigen::VectorXd &)> log_likelihood_function,
	std::function<double(Eigen::VectorXd &)> log_prior_function,
	bool return_samples
)
{
	return make_unique<OneKernelResult>(
		_apply_one_kernel_rwm(sample, n_transitions, sigma, log_likelihood_function, log_prior_function, return_samples)
	);
}

std::unique_ptr<KernelResult> apply_kernel_rwm(
	const Eigen::Ref<const Eigen::MatrixXd> &samples,
	const int n_transitions,
	const Eigen::VectorXd &sigma,
	std::function<double(Eigen::VectorXd &)> log_likelihood_function,
	std::function<double(Eigen::VectorXd &)> log_prior_function,
	const int n_threads
)
{
	DEBUG_FUNCTION_ENTER;
	LOG_TRACE(n_transitions << " transitions.");
	LOG_TRACE("Samples has " << samples.rows() << " rows and " << samples.cols() << " columns.");

	Eigen::VectorXd acceptances(samples.rows());
	Eigen::VectorXd log_likelihoods(samples.rows());
	Eigen::VectorXd log_targets(samples.rows());
	Eigen::MatrixXd results(samples.rows(), samples.cols());

	Eigen::setNbThreads(1);
	const int n_iter = samples.rows();

	#pragma omp parallel for num_threads(n_threads)
	for(int i = 0; i < n_iter; i++)
	{
		Eigen::VectorXd sample = samples.row(i);
		OneKernelResult res = _apply_one_kernel_rwm(sample, n_transitions, sigma, log_likelihood_function, log_prior_function, false);
		LOG_TRACE("Output contains " << res.result);
		
		results.row(i) = res.result;
		LOG_TRACE("Results row " << i << " contains " << results.row(i));
		acceptances(i) = res.average_acceptance;
		log_likelihoods(i) = res.log_likelihood;
		log_targets(i) = res.log_target;
	}
	LOG_TRACE("finished apply_kernel_rwm, results are " << results);
	auto ret = make_unique<KernelResult>(results, log_likelihoods, log_targets, acceptances);
	LOG_TRACE("KernelResult contains " << (ret->results));
	LOG_TRACE("And now I'm just testing that results is still here " << results);
	
	DEBUG_FUNCTION_EXIT;
	return ret;
}
