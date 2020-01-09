#include "operators.hpp"
#include "collocate.hpp"
#include <Eigen/Dense>
#include "memory_utils.hpp"
#include <iostream>
#include <fstream>


Collocator::Collocator(const Eigen::Ref<const Eigen::MatrixXd> &x, int N_collocate, const Eigen::Ref<const Eigen::VectorXd> &kernel_args) {
	_kern = Id_Id(x, x, kernel_args);
	_left = Eigen::MatrixXd(x.rows(), N_collocate);
	_central = Eigen::MatrixXd(N_collocate, N_collocate);
}

std::unique_ptr<CollocationResult> Collocator::collocate_no_obs(
	const Eigen::Ref<const Eigen::MatrixXd> &x,
	const Eigen::Ref<const Eigen::MatrixXd> &interior, 
	const Eigen::Ref<const Eigen::MatrixXd> &sensors,
	const Eigen::Ref<const Eigen::VectorXd> &kernel_args
)
{
	Id_A(x, interior, kernel_args, _left.leftCols(interior.rows()));
	Id_B(x, sensors, kernel_args, _left.rightCols(sensors.rows()));

	A_A(interior, interior, kernel_args, _central.topLeftCorner(interior.rows(), interior.rows()));
	A_B(interior, sensors, kernel_args, _central.topRightCorner(interior.rows(), sensors.rows()));
	B_B(sensors, sensors, kernel_args, _central.bottomRightCorner(sensors.rows(), sensors.rows()));
	_central.bottomLeftCorner(sensors.rows(), interior.rows()) = _central.topRightCorner(interior.rows(), sensors.rows()).transpose();

	//Id_Id(x, x, kernel_args, kern);

	/*
	std::ofstream file1("central.txt");
	file1 << central;
	std::ofstream file2("left.txt");
	file2 << left;
	std::ofstream file3("kern.txt");
	file3 << kern;
	*/
	// and lastly build the posterior
	// first invert the central matrix...
	Eigen::MatrixXd tmp = _central.ldlt().solve(_left.transpose());
	Eigen::MatrixXd mu_mult = tmp.transpose();
	Eigen::MatrixXd cov = _kern - mu_mult * _left.transpose();

	return make_unique<CollocationResult>(mu_mult, cov);
}

std::unique_ptr<CollocationResult> collocate_no_obs(
	const Eigen::Ref<const Eigen::MatrixXd> &x,
	const Eigen::Ref<const Eigen::MatrixXd> &interior, 
	const Eigen::Ref<const Eigen::MatrixXd> &sensors,
	const Eigen::Ref<const Eigen::VectorXd> &kernel_args
)
{
	Collocator collocator(x, interior.rows() + sensors.rows(), kernel_args);
	return collocator.collocate_no_obs(x, interior, sensors, kernel_args);
}