#include <limits>
#include "potentials.hpp"
#include "logging/logging.hpp"
double log_likelihood_indicator(double norm, double delta) {
	if(norm > delta)
		return -1*std::numeric_limits<double>::infinity();
    return 0.0;
}

double log_likelihood_gaussian(
	const Eigen::Ref<const Eigen::VectorXd> &x, 
	const Eigen::Ref<const Eigen::VectorXd> &mean, 
	const Eigen::Ref<const Eigen::VectorXd> &sigma
) {
	double ret = 0.;
	for(int i = 0; i < x.rows(); i++)
	{
		double centered = x(i) - mean(i);
		double sigma2 = sigma(i)*sigma(i);
		ret += centered*centered / sigma2;
	}
	return -0.5*ret;
}

double log_likelihood_gaussian(
		const Eigen::Ref<const Eigen::VectorXd> &x,
		const Eigen::Ref<const Eigen::VectorXd> &mean,
		double sigma
) {
	double ret = 0.;
	for(int i = 0; i < x.rows(); i++)
	{
		double centered = x(i) - mean(i);
		ret += centered*centered;
	}
	return -0.5*ret / (sigma*sigma);
}