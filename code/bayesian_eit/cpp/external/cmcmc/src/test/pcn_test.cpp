#include <unsupported/Eigen/MatrixFunctions>
#include "pcn.hpp"
#include "catch.hpp"

Eigen::MatrixXd make_sqexp(const Eigen::VectorXd &x, const Eigen::VectorXd &y, double length_scale) {
	Eigen::MatrixXd ret(x.rows(), y.rows());
	double denom = 2*length_scale*length_scale;
	for(int i = 0; i < x.rows(); i++) {
		double x_i = x(i);
		for(int j = 0; j < y.rows(); j++) {
			double delta = x_i - y(j);
			ret(i, j) = exp(-delta*delta / denom);
		}
	}
	return ret;
}

TEST_CASE("Simple_PCN") {
	int n_rows = 11;
	int n_iter = 10000;
	// simple 1D example
	double mult = 2./(n_rows-1);
	Eigen::VectorXd x(n_rows);
	for(int i = 0; i <  n_rows; i++) 
		x(i) = i*mult - 1;
	Eigen::MatrixXd cov = make_sqexp(x,x,0.3);
	Eigen::MatrixXd sqrt_cov = cov.sqrt();
	Eigen::VectorXd ones = Eigen::VectorXd::Ones(n_rows);

	Eigen::VectorXd target = -1.0*(x-ones).cwiseProduct(x+ones);
	double likelihood_noise = 0.01;

	CAPTURE(x);
	CAPTURE(target);

	auto log_likelihood = [target, likelihood_noise] (const Eigen::VectorXd &x) -> double {
		Eigen::VectorXd delta = target - x;
		return -delta.dot(delta) / (2*likelihood_noise*likelihood_noise);
	};

	Eigen::VectorXd x0 = Eigen::VectorXd::Zero(n_rows);

	auto results = apply_one_kernel_pcn(x0, n_iter, 0.05, log_likelihood, sqrt_cov, true);
	Eigen::VectorXd mean(n_rows);
	CAPTURE(results->average_acceptance);
	for(int i = 0; i < n_rows; i++) {
		mean(i) = results->samples.col(i).tail(int(n_iter*0.5)).mean();
		CHECK(mean(i) == Approx(target(i)).epsilon(2*likelihood_noise));
	}
}
