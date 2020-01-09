#include <iostream>
#include <random>
#include <memory>
#include "mala.hpp"
#include "debug.hpp"
#include <math.h>
#include "memory_utils.hpp"
#include <future>

struct TransitionProbabilities {
	double cur_to_new;
	double new_to_cur;
};

TransitionProbabilities calc_q(
	const Eigen::VectorXd &x_cur, 
	const Eigen::VectorXd &x_new, 
	const Eigen::VectorXd &grad_cur, 
	const Eigen::VectorXd &grad_new, 
	double tau, 
	const Eigen::VectorXd &preconditioner) 
{
	double q_num = 0;
	double q_denom = 0;
	double tmp1, tmp2;
	for(int i = 0; i < x_cur.rows(); i++) {
		tmp1 = x_cur(i) - x_new(i) - tau*preconditioner(i)*grad_new(i);
		tmp2 = 4*tau*preconditioner(i);
		q_num -= tmp1*tmp1/tmp2;
		tmp1 = x_new(i) - x_cur(i) - tau*preconditioner(i)*grad_cur(i);
		q_denom -= tmp1*tmp1/tmp2;
	}
	TransitionProbabilities ret;
	ret.new_to_cur = q_num;
	ret.cur_to_new = q_denom;

	return ret;
}

OneKernelResult _apply_one_kernel_mala(
	const Eigen::VectorXd &sample,
	int n_transitions,
	double tau,
	std::function<GradientEval(Eigen::VectorXd &)> log_likelihood_function,
	std::function<GradientEval(Eigen::VectorXd &)> log_prior_function,
	const Eigen::VectorXd &preconditioner,
	bool return_samples
)
{
	double NEGINF = -1*std::numeric_limits<double>::infinity();
	DEBUG_FUNCTION_ENTER;
	LOG_TRACE("Sample has " << sample.rows() << " rows");

    double accept_ratio;
    bool accept;
	Eigen::VectorXd cur = sample;
	Eigen::VectorXd next(cur.rows());
	
	GradientEval cur_log_likelihood = log_likelihood_function(cur);
	GradientEval cur_log_prior = log_prior_function(cur);

	LOG_TRACE("Initial log-likelihood is " << cur_log_likelihood.value);

	std::default_random_engine generator(std::random_device{}());
	std::uniform_real_distribution<double> uniform_distribution(0.0,1.0);

	Eigen::MatrixXd ret_samples;
	Eigen::VectorXi acceptances;
	if(return_samples) {
		ret_samples = Eigen::MatrixXd(n_transitions, sample.rows());
		acceptances = Eigen::VectorXi(n_transitions);
	}

	int accepts = 0;
	LOG_TRACE("About to start " << n_transitions << " MALA transitions with tau=" << tau);
	for(int i = 0; i < n_transitions; i++) {
		// this is shit but for some reason can't pass RNGs about...
		for(int j = 0; j < sample.rows(); j++) {
			std::normal_distribution<double> distribution(0.0, 1.);
			double random_component = sqrt(2*tau*preconditioner(j))*distribution(generator);
			double grad_component = tau*preconditioner(j)*(cur_log_likelihood.gradient(j) + cur_log_prior.gradient(j));
			next(j) = cur(j) + grad_component + random_component;
		}

		GradientEval new_log_likelihood = log_likelihood_function(next);
		GradientEval new_log_prior = log_prior_function(next);

		LOG_TRACE("Iteration " << i << ": proposal done; it was " << next << " and has log-likelihood " << new_log_likelihood.value << ".");
		accept = false;
		// new sample is not in the support of the prior, reject
		if(new_log_prior.value == NEGINF) {
			LOG_TRACE("Not in support of prior; continuing.");
		}
		else if(new_log_likelihood.value == NEGINF) {
			LOG_TRACE("Potential is negative infinity; continuing.");
		}
		else if(cur_log_likelihood.value == NEGINF) {
			accept = true;
			LOG_TRACE("Old potential was neginf but new is not; accepting.");
		}
		// both current and new sample are in the ball, reject with probability based on the prior ratio.
		else {
			TransitionProbabilities probs = calc_q(
				cur, 
				next, 
				cur_log_likelihood.gradient, 
				new_log_likelihood.gradient, 
				tau, 
				preconditioner
			);
			Eigen::VectorXd denom = -1./(4*tau*preconditioner.array());

			double q_component = probs.new_to_cur - probs.cur_to_new;
			double prior_component = new_log_prior.value - cur_log_prior.value;
			double likelihood_component = new_log_likelihood.value - cur_log_likelihood.value;

            accept_ratio = exp(likelihood_component + prior_component + q_component);
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
		cur_log_likelihood.value,
		cur_log_likelihood.value + cur_log_prior.value,
		accepts * 1.0/((double)n_transitions));
	LOG_TRACE("Result contains " << res.result);
	
	DEBUG_FUNCTION_EXIT;
	return res;
}


std::unique_ptr<OneKernelResult> apply_one_kernel_mala(
	const Eigen::VectorXd &sample,
	int n_transitions,
	const double tau,
	std::function<GradientEval(Eigen::VectorXd &)> log_likelihood_function,
	std::function<GradientEval(Eigen::VectorXd &)> log_prior_function,
	const Eigen::VectorXd &preconditioner,
	bool return_samples
)
{
	return make_unique<OneKernelResult>(
		_apply_one_kernel_mala(sample, 
			n_transitions, 
			tau, 
			log_likelihood_function, 
			log_prior_function, 
			preconditioner, 
			return_samples)
	);
}

std::unique_ptr<KernelResult> apply_kernel_mala(
	const Eigen::Ref<const Eigen::MatrixXd> &samples,
	const int n_transitions,
	const double tau,
	std::function<GradientEval(Eigen::VectorXd &)> log_likelihood_function,
	std::function<GradientEval(Eigen::VectorXd &)> log_prior_function,
	const Eigen::VectorXd &preconditioner,
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
		OneKernelResult res = _apply_one_kernel_mala(
			sample, 
			n_transitions, 
			tau, 
			log_likelihood_function, 
			log_prior_function, 
			preconditioner, 
			false
		);
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
