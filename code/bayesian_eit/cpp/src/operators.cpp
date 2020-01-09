#include <Eigen/Core>
#include "operators.hpp"

Eigen::MatrixXd Id_Id(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args) {
    Eigen::MatrixXd ret(xarg.rows(), yarg.rows());
    Id_Id(xarg, yarg, args, ret);
    return ret;
}

void Id_Id(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args, Eigen::Ref<Eigen::MatrixXd> ret) {
    double l = args(0);
    double sigma = args(1);

    double l2 = l*l;
    double l2mul = 1./l2;
    for(int i = 0; i < xarg.rows(); i++) {
        double x = xarg(i, 0);
        double y = xarg(i, 1);

        for(int j = 0; j < yarg.rows(); j++) {
            double xbar = yarg(j, 0);
            double ybar = yarg(j, 1);

            double dx = x-xbar;
            double dy = y-ybar;
            double dx2 = dx*dx;
            double dy2 = dy*dy;
            double kval = exp(0.5*(-dx2 - dy2)*l2mul);
            ret(i, j) = sigma*kval;
        }
    }
}

Eigen::MatrixXd Id_A(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args) {
    Eigen::MatrixXd ret(xarg.rows(), yarg.rows());
    Id_A(xarg, yarg, args, ret);
    return ret;
}

void Id_A(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args, Eigen::Ref<Eigen::MatrixXd> ret) {
    double l = args(0);
    double sigma = args(1);

    
    double l2 = l*l;
    double l2mul = 1./l2;
    for(int i = 0; i < xarg.rows(); i++) {
        double x = xarg(i, 0);
        double y = xarg(i, 1);
        double a = xarg(i, 2);
        double a_x = xarg(i, 3);
        double a_y = xarg(i, 4);

        for(int j = 0; j < yarg.rows(); j++) {
            double xbar = yarg(j, 0);
            double ybar = yarg(j, 1);
            double abar = yarg(j, 2);
            double a_xbar = yarg(j, 3);
            double a_ybar = yarg(j, 4);

            
            double dx = x-xbar;
            double dy = y-ybar;
            double dx2 = dx*dx;
            double dy2 = dy*dy;
            double kval = exp(0.5*(-dx2 - dy2)*l2mul);
            ret(i, j) = sigma*(a_xbar*dx + a_ybar*dy + (-1.0 + dx2*l2mul) + (-1.0 + dy2*l2mul))*kval*l2mul*exp(abar);
        }
    }
}


Eigen::MatrixXd Id_B(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args) {
    Eigen::MatrixXd ret(xarg.rows(), yarg.rows());
    Id_B(xarg, yarg, args, ret);
    return ret;
}

void Id_B(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args, Eigen::Ref<Eigen::MatrixXd> ret) {
    double l = args(0);
    double sigma = args(1);

    
    double l2 = l*l;
    double l2mul = 1./l2;
    for(int i = 0; i < xarg.rows(); i++) {
        double x = xarg(i, 0);
        double y = xarg(i, 1);
        double a = xarg(i, 2);
        double a_x = xarg(i, 3);
        double a_y = xarg(i, 4);

        for(int j = 0; j < yarg.rows(); j++) {
            double xbar = yarg(j, 0);
            double ybar = yarg(j, 1);
            double abar = yarg(j, 2);
            double a_xbar = yarg(j, 3);
            double a_ybar = yarg(j, 4);

            
            double dx = x-xbar;
            double dy = y-ybar;
            double dx2 = dx*dx;
            double dy2 = dy*dy;
            double kval = exp(0.5*(-dx2 - dy2)*l2mul);
            ret(i, j) = sigma*(xbar*dx + ybar*dy)*exp(abar)*kval*l2mul;
        }
    }
}


Eigen::MatrixXd A_A(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args) {
    Eigen::MatrixXd ret(xarg.rows(), yarg.rows());
    A_A(xarg, yarg, args, ret);
    return ret;
}

void A_A(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args, Eigen::Ref<Eigen::MatrixXd> ret) {
    double l = args(0);
    double sigma = args(1);

    
    double l2 = l*l;
    double l2mul = 1./l2;
    for(int i = 0; i < xarg.rows(); i++) {
        double x = xarg(i, 0);
        double y = xarg(i, 1);
        double a = xarg(i, 2);
        double a_x = xarg(i, 3);
        double a_y = xarg(i, 4);

        for(int j = 0; j < yarg.rows(); j++) {
            double xbar = yarg(j, 0);
            double ybar = yarg(j, 1);
            double abar = yarg(j, 2);
            double a_xbar = yarg(j, 3);
            double a_ybar = yarg(j, 4);

            
            double dx = x-xbar;
            double dy = y-ybar;
            double dx2 = dx*dx;
            double dy2 = dy*dy;
            double dxdy = dx*dy;
            double dx3 = dx*dx2;
            double dy3 = dy*dy2;

            double dx2_l2 = dx2*l2mul;
            double dy2_l2 = dy2*l2mul;
            double dxdy_l2 = dxdy*l2mul;
            double kval = exp(0.5*(-dx2 - dy2)*l2mul);
            ret(i, j) = sigma*exp(a)*exp(abar)*kval*l2mul*(
                a_x*(a_xbar*(1 - dx2_l2) - a_ybar*dxdy_l2 - (dx2_l2 + dy2_l2 - 4.0)*dx*l2mul) 
                + a_y*(a_ybar*(1 - dy2_l2) - a_xbar*dxdy_l2 - (dx2_l2 + dy2_l2 - 4.0)*dy*l2mul)
                - l2mul*(a_xbar*dx*(1.0 - dy2_l2) + a_ybar*dy*(3.0 - dy2_l2) - 4.0 + dy2_l2*(7.0 - dx2_l2 - dy2_l2) + dx2_l2) 
                - l2mul*(a_xbar*dx*(3.0 - dx2_l2) + a_ybar*dy*(1.0 - dx2_l2) - 4.0 + dx2_l2*(7.0 - dx2_l2 - dy2_l2) + dy2_l2)
            );
        }
    }
}

Eigen::MatrixXd A_B(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args) {
    Eigen::MatrixXd ret(xarg.rows(), yarg.rows());
    A_B(xarg, yarg, args, ret);
    return ret;
}

void A_B(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args, Eigen::Ref<Eigen::MatrixXd> ret) {
    double l = args(0);
    double sigma = args(1);

    
    double l2 = l*l;
    double l2mul = 1./l2;
    for(int i = 0; i < xarg.rows(); i++) {
        double x = xarg(i, 0);
        double y = xarg(i, 1);
        double a = xarg(i, 2);
        double a_x = xarg(i, 3);
        double a_y = xarg(i, 4);

        for(int j = 0; j < yarg.rows(); j++) {
            double xbar = yarg(j, 0);
            double ybar = yarg(j, 1);
            double abar = yarg(j, 2);
            double a_xbar = yarg(j, 3);
            double a_ybar = yarg(j, 4);

            
            double dx = x-xbar;
            double dy = y-ybar;
            double dx2 = dx*dx;
            double dy2 = dy*dy;
            double dxdy = dx*dy;
            double kval = exp(0.5*(-dx2 - dy2)*l2mul);
            /*
            ret(i, j) = sigma*exp(a)*exp(abar)*kval*l2mul*(
                a_x*(xbar*(1 - dx2*l2mul) - ybar*dxdy*l2mul) 
                + a_y*(ybar*(1 - ybar*dy2*l2mul) - xbar*dxdy*l2mul) 
                - 1.*l2mul*(xbar*dx + 3.0*ybar*dy - dy2*(xbar*dx*l2mul + ybar*dy*l2mul))
                - 1.*l2mul*(3.0*xbar*dx + ybar*dy - dx2*(xbar*dx*l2mul + ybar*dy*l2mul))
            );
            */
            ret(i, j) = exp(a)*exp(abar)*kval*sigma*l2mul*(
                a_x*(
                    xbar
                    - xbar*dx2*l2mul
                    - ybar*dxdy*l2mul
                )
                + a_y*(
                    ybar
                    - xbar*dxdy*l2mul 
                    - ybar*dy2*l2mul
                ) 
                - l2mul*(
                    xbar*dx 
                    + 3.0*ybar*dy 
                    - dy2*(
                        xbar*dx*l2mul 
                        + ybar*dy*l2mul
                    )
                )
                - l2mul*(
                    3.0*xbar*dx 
                    + ybar*dy 
                    - dx2*(
                        xbar*dx*l2mul 
                        + ybar*dy*l2mul
                    )
                )
            );
        }
    }
}

Eigen::MatrixXd B_B(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args) {
    Eigen::MatrixXd ret(xarg.rows(), yarg.rows());
    B_B(xarg, yarg, args, ret);
    return ret;
}

void B_B(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args, Eigen::Ref<Eigen::MatrixXd> ret) {
    double l = args(0);
    double sigma = args(1);

    double l2 = l*l;
    double l2mul = 1./l2;
    for(int i = 0; i < xarg.rows(); i++) {
        double x = xarg(i, 0);
        double y = xarg(i, 1);
        double a = xarg(i, 2);
        double a_x = xarg(i, 3);
        double a_y = xarg(i, 4);

        for(int j = 0; j < yarg.rows(); j++) {
            double xbar = yarg(j, 0);
            double ybar = yarg(j, 1);
            double abar = yarg(j, 2);
            double a_xbar = yarg(j, 3);
            double a_ybar = yarg(j, 4);

            double dx = x-xbar;
            double dy = y-ybar;
            double dx2 = dx*dx;
            double dy2 = dy*dy;
            double dxdy = dx*dy;
            double kval = exp(0.5*(-dx2 - dy2)*l2mul);
            ret(i, j) = exp(a)*exp(abar)*sigma*kval*l2mul*(
                x*(xbar*(1 - dx2*l2mul) - ybar*dxdy*l2mul) 
                + y*(ybar*(1 - dy2*l2mul) - xbar*dxdy*l2mul)
            );
        }
    }
}