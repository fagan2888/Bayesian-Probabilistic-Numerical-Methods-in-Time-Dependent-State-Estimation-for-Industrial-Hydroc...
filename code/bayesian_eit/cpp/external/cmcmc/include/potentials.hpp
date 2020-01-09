#include <Eigen/Core>
double log_likelihood_indicator(double norm, double delta);
double log_likelihood_gaussian(
	const Eigen::Ref<const Eigen::VectorXd> &x, 
	const Eigen::Ref<const Eigen::VectorXd> &mean, 
	const Eigen::Ref<const Eigen::VectorXd> &sigma
);
double log_likelihood_gaussian(
		const Eigen::Ref<const Eigen::VectorXd> &x,
		const Eigen::Ref<const Eigen::VectorXd> &mean,
		double sigma
);
