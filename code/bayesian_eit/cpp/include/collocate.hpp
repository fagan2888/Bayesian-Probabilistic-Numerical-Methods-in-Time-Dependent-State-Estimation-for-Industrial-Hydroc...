#include <Eigen/Core>
#include <memory>

#ifndef COLLOCATE_H

class CollocationResult {
public:
	const Eigen::MatrixXd mu_mult;
	const Eigen::MatrixXd cov;
	CollocationResult(Eigen::MatrixXd mu_mult, Eigen::MatrixXd cov) : mu_mult(mu_mult), cov(cov) { };
};

class Collocator {
public:
	Collocator(const Eigen::Ref<const Eigen::MatrixXd> &x, int N_collocate, const Eigen::Ref<const Eigen::VectorXd> &kernel_args);
	std::unique_ptr<CollocationResult> collocate_no_obs(
		const Eigen::Ref<const Eigen::MatrixXd> &x,
		const Eigen::Ref<const Eigen::MatrixXd> &interior, 
		const Eigen::Ref<const Eigen::MatrixXd> &sensors,
		const Eigen::Ref<const Eigen::VectorXd> &kernel_args
	);
private:
	Eigen::MatrixXd _left;
	Eigen::MatrixXd _central;
	Eigen::MatrixXd _kern;
};

std::unique_ptr<CollocationResult> collocate_no_obs(
	const Eigen::Ref<const Eigen::MatrixXd> &x,
	const Eigen::Ref<const Eigen::MatrixXd> &interior, 
	const Eigen::Ref<const Eigen::MatrixXd> &sensors,
	const Eigen::Ref<const Eigen::VectorXd> &kernel_args
);

#define COLLOCATE_H
#endif