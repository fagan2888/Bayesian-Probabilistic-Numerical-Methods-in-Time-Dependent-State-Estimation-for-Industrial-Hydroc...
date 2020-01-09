#include<Eigen/Core>
#ifndef OPERATORS_H

Eigen::MatrixXd Id_Id(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args);
Eigen::MatrixXd Id_A(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args);
Eigen::MatrixXd Id_B(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args);
Eigen::MatrixXd A_A(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args);
Eigen::MatrixXd A_B(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args);
Eigen::MatrixXd B_B(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args);

void Id_Id(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args, Eigen::Ref<Eigen::MatrixXd> ret);
void Id_A(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args, Eigen::Ref<Eigen::MatrixXd> ret);
void Id_B(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args, Eigen::Ref<Eigen::MatrixXd> ret);
void A_A(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args, Eigen::Ref<Eigen::MatrixXd> ret);
void A_B(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args, Eigen::Ref<Eigen::MatrixXd> ret);
void B_B(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args, Eigen::Ref<Eigen::MatrixXd> ret);

#define OPERATORS_H
#endif