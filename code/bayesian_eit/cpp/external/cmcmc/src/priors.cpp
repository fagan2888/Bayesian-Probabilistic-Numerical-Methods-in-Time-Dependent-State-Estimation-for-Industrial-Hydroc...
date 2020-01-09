#include "priors.hpp"
#include "debug.hpp"


double eval_log_prior(PriorType type, const Eigen::VectorXd &sample, const Eigen::VectorXd &loc, const Eigen::VectorXd &scale) {
	if(type == PriorType::GAUSSIAN)
		return log_gaussian_dist(sample, loc, scale);
	if(type == PriorType::CAUCHY)
		return log_cauchy_dist(sample, loc, scale);
	if(type == PriorType::UNIFORM)
		return log_uniform_dist(sample, loc, scale);
	throw "PriorType not understood!";
}

GradientEval eval_grad_log_prior(PriorType type, const Eigen::VectorXd &sample, const Eigen::VectorXd &loc, const Eigen::VectorXd &scale) {
	double prior_eval = eval_log_prior(type, sample, loc, scale);

	if(type == PriorType::GAUSSIAN)
		return GradientEval(prior_eval, grad_log_gaussian_dist(sample, loc, scale));
    if(type == PriorType::CAUCHY)
        return GradientEval(prior_eval, grad_log_cauchy_dist(sample, loc, scale));
	if(type == PriorType::UNIFORM)
		return GradientEval(prior_eval, Eigen::VectorXd::Zero(sample.rows()));

	throw "PriorType not understood!";
}

double log_cauchy_dist(const Eigen::VectorXd &sample, const Eigen::VectorXd &x_0, const Eigen::VectorXd &gamma) {
	DEBUG_FUNCTION_ENTER;
	double ret = 0;
	double tmp_sq;
    for(int i = 0; i < sample.rows(); i++) {	
    	tmp_sq = (sample(i) - x_0(i)) / gamma(i);
    	ret += -log(gamma(i)) - log(1 + tmp_sq*tmp_sq);
    }
    DEBUG_FUNCTION_EXIT;
    return ret;
}

Eigen::VectorXd grad_log_cauchy_dist(const Eigen::VectorXd &sample, const Eigen::VectorXd &x_0, const Eigen::VectorXd &gamma) {
	DEBUG_FUNCTION_ENTER;
    double tmp_sq = 0;
    double gamma_val = 0;
    Eigen::VectorXd ret(sample.rows());
    for(int i = 0; i < sample.rows(); i++) {
        gamma_val = gamma(i);
        tmp_sq = (sample(i) - x_0(i)) / gamma_val;
        ret(i) = -2*tmp_sq / (gamma_val*(1+tmp_sq*tmp_sq));
    }
    DEBUG_FUNCTION_EXIT;
    return ret;
}

double log_gaussian_dist(const Eigen::VectorXd &sample, const Eigen::VectorXd &loc, const Eigen::VectorXd &scale) {
	DEBUG_FUNCTION_ENTER;
	double ret = 0;
	double tmp_sq;
	for(int i = 0; i < sample.rows(); i++) {
		tmp_sq = (sample(i) - loc(i));
		ret -= tmp_sq*tmp_sq / (2.*scale(i)*scale(i));
	}

	DEBUG_FUNCTION_EXIT;
	return ret;
}

Eigen::VectorXd grad_log_gaussian_dist(const Eigen::VectorXd &sample, const Eigen::VectorXd &loc, const Eigen::VectorXd &scale) {
	DEBUG_FUNCTION_ENTER;
	Eigen::VectorXd ret(sample.rows());
	double tmp_sq;
	for(int i = 0; i < sample.rows(); i++) {
		tmp_sq = (sample(i) - loc(i));
		ret(i) = -tmp_sq / (scale(i)*scale(i));
	}

	DEBUG_FUNCTION_EXIT;
	return ret;
}

double log_uniform_dist(const Eigen::VectorXd &sample, const Eigen::VectorXd &loc, const Eigen::VectorXd &scale) {
	DEBUG_FUNCTION_ENTER;
	double NEGINF = -1*std::numeric_limits<double>::infinity();
	for(int i = 0; i < sample.rows(); i++) {
		double samp = sample(i);
		double center = loc(i);
		double width = scale(i);
		if(samp > center + width || samp < center - width) {
			DEBUG_FUNCTION_EXIT;
			return NEGINF;
		}
	}

	DEBUG_FUNCTION_EXIT;
	return 0.0;
}