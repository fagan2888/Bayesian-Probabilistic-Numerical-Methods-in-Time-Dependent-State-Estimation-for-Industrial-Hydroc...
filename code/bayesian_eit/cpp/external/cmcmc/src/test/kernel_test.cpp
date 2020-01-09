#include <Eigen/Core>
#include <iostream>
#include "rwm.hpp"
#include "catch.hpp"
#include "kernels/rwm_kernel.hpp"

TEST_CASE("Simple 1D Gaussian RWM Kernel") {
	double mean = 2.0;
	double var = 0.1;

	double prior_mean = 0.0;
	double prior_var = 4.0;

	auto log_likelihood_function = [mean, var] (const Eigen::VectorXd &x) -> double {
    	return -0.5*(x(0) - mean)*(x(0) - mean) / var;
    };

    auto log_prior = [prior_mean, prior_var] (const Eigen::VectorXd &x) -> double {
    	return -0.5*(x(0) - prior_mean)*(x(0) - prior_mean) / prior_var;
    };

    std::srand(0);
    auto prior_samples = 2*Eigen::MatrixXd::Random(100, 1).array() + 2.0;
    auto proposal_var = Eigen::VectorXd::Ones(1);

    std::unique_ptr<KernelResult> res = apply_kernel_rwm(
		prior_samples,
		10000,
		proposal_var,
		log_likelihood_function,
		log_prior,
		1);
    CHECK(res->results.mean() == Approx(mean).epsilon(1e-1));

    for(int i = 0; i < res->results.rows(); i++)
    {
    	Eigen::VectorXd sample = res->results.row(i);
    	double likelihood = log_likelihood_function(sample);
    	CHECK(res->log_likelihoods(i) == likelihood);
    }
}

TEST_CASE("Simple 1D Gaussian RWM Kernel OO") {
    double mean = 2.0;
    double var = 0.1;

    double prior_mean = 0.0;
    double prior_var = 4.0;

    auto log_likelihood_function = [mean, var] (const Eigen::VectorXd &x) -> double {
        return -0.5*(x(0) - mean)*(x(0) - mean) / var;
    };

    auto log_prior = [prior_mean, prior_var] (const Eigen::VectorXd &x) -> double {
        return -0.5*(x(0) - prior_mean)*(x(0) - prior_mean) / prior_var;
    };

    std::srand(0);
    auto prior_sample = 2*Eigen::VectorXd::Random(1).array() + 2.0;
    auto proposal_var = Eigen::VectorXd::Ones(1);

    TransitionKernel *kernel = new RWMKernel(log_likelihood_function, log_prior, proposal_var);
    auto result = kernel->apply(prior_sample, 10000, true);
    CHECK(result->samples.mean() == Approx(mean).epsilon(1e-1));
    double likelihood = log_likelihood_function(result->result);
    CHECK(result->log_likelihood == likelihood);
}

TEST_CASE("Simple 2D Gaussian RWM Kernel") {
	Eigen::VectorXd mean = 2*Eigen::VectorXd::Ones(2);
	Eigen::VectorXd var = 0.1*Eigen::VectorXd::Ones(2);

	Eigen::VectorXd prior_mean = Eigen::VectorXd::Zero(2);
	Eigen::VectorXd prior_var = 4*Eigen::VectorXd::Ones(2);

	auto log_likelihood_function = [mean, var] (const Eigen::VectorXd &x) -> double {
    	return -0.5*((x - mean).cwiseProduct(x-mean).array() / var.array()).sum();
    };

    auto log_prior = [prior_mean, prior_var] (const Eigen::VectorXd &x) -> double {
    	return -0.5*((x - prior_mean).cwiseProduct(x-prior_mean).array() / prior_var.array()).sum();
    };

    std::srand(0);
    auto prior_samples = 2*Eigen::MatrixXd::Random(100, 2).array() + 2.0;
    auto proposal_var = Eigen::VectorXd::Ones(2);

    std::unique_ptr<KernelResult> res = apply_kernel_rwm(
		prior_samples,
		10000,
		proposal_var,
		log_likelihood_function,
		log_prior,
		1);
    CHECK(res->results.col(0).mean() == Approx(mean(0)).epsilon(1e-1));
    CHECK(res->results.col(1).mean() == Approx(mean(1)).epsilon(1e-1));
    for(int i = 0; i < res->results.rows(); i++)
    {
    	Eigen::VectorXd sample = res->results.row(i);
    	double likelihood = log_likelihood_function(sample);
    	CHECK(res->log_likelihoods(i) == likelihood);
    }
}