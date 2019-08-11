//
// Created by lancelot on 3/15/17.
//

#include <glog/logging.h>

#include "ICPSimulation.h"


int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    Eigen::Matrix<double, 6, 1> xi;
    xi << 23.0, 11.0, 12.0, 0.7, 1.2, 0.33;
    Eigen::Matrix3d Var = Eigen::Matrix3d::Zero();
    Var(0, 0) = 1;
    Var(1, 1) = 3;
    Var(1, 0) = Var(0, 1) = 1.5;
	Var(2, 2) = 2;

    Sophus::SE3d T = Sophus::SE3d::exp(xi);
    //std::cout << T.matrix3x4() << std::endl;
    ICPSimulation simulation(T, Var);
    simulation.start();


    return 0;
}