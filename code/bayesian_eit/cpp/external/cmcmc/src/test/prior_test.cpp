#include "catch.hpp"
#include "priors.hpp"
#include <Eigen/Core>

TEST_CASE("Negative infinity does what it should do when exped", "neginf_test") {
	Eigen::VectorXd samp = Eigen::VectorXd(2);
	Eigen::VectorXd loc = Eigen::VectorXd(2);
	Eigen::VectorXd scale = Eigen::VectorXd(2);

	samp << 1.,1.;
	loc << 0., 0.;
	scale << 1.5, 0.5;

	double ret = log_uniform_dist(samp, loc, scale);

	CHECK(ret == -std::numeric_limits<double>::infinity());
	CHECK(exp(ret) == 0);
}

TEST_CASE("Gaussian log gradient is correct") {
	for(int i = 0; i < 100; i++) {
		Eigen::VectorXd loc = Eigen::VectorXd::Random(2);
		Eigen::VectorXd scale = Eigen::VectorXd::Random(2);
		Eigen::VectorXd samp = Eigen::VectorXd::Random(2);

		Eigen::VectorXd gradient = grad_log_gaussian_dist(samp, loc, scale);
		double h = 1e-8;
		for(int j = 0; j < samp.rows(); j++) {
			Eigen::VectorXd up = samp;
			Eigen::VectorXd down = samp;
			up(j) += h;
			down(j) -= h;

			double approx_deriv = (log_gaussian_dist(up, loc, scale) - log_gaussian_dist(down, loc, scale)) / (2.*h);
			CHECK(gradient(j) == Approx(approx_deriv));
		}
	}
}

TEST_CASE("Cauchy log gradient is correct") {
	for(int i = 0; i < 100; i++) {
		Eigen::VectorXd loc = Eigen::VectorXd::Random(2);
		Eigen::VectorXd scale = Eigen::VectorXd::Random(2).array().abs();
		Eigen::VectorXd samp = Eigen::VectorXd::Random(2);

		Eigen::VectorXd gradient = grad_log_cauchy_dist(samp, loc, scale);
		double h = 1e-8;
		for(int j = 0; j < samp.rows(); j++) {
			Eigen::VectorXd up = samp;
			Eigen::VectorXd down = samp;
			up(j) += h;
			down(j) -= h;

			double approx_deriv = (log_cauchy_dist(up, loc, scale) - log_cauchy_dist(down, loc, scale)) / (2.*h);
			CHECK(gradient(j) == Approx(approx_deriv));
		}
	}
}