// 参考https://www.jianshu.com/p/e5b03cf22c80

#include <iostream>
#include <ceres/ceres.h>

using namespace std;
using namespace ceres;

/*
仿函数（functor）的英文解释为something that performs a function，即其行为类似函数的东西。
C++中的仿函数是通过在类中重载()运算符实现，使你可以像使用函数一样来创建类的对象。
 */

//第一部分：构建代价函数，重载（）符号，仿函数的小技巧
struct CostFunctor{
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        residual[0] = T(10.0)-x[0];
        return true;
    }
};

int main(int argc, char** argv) {

    google::InitGoogleLogging(argv[0]);

    //寻优参数x的初始值，为5
    double initial_x = 1000000.0;
    double x = initial_x;


    //第二部分：构建残差方程
    Problem problem;
    //使用自动求导，将之前的代价函数结构体传入，第一个1是输出维度，即残差的维度，第二个1是输入维度，即待寻优参数x的维度。
    CostFunction* cost_function = new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);

    //向问题中添加误差项，本问题比较简单，添加一个就行。这里的参数NULL是指不使用核函数，&x表示x是待寻优参数。
    problem.AddResidualBlock(cost_function,NULL,&x);


    //第三部分： 配置并运行求解器
    Solver::Options options;
    //配置增量方程的解法,有DENSE_QR、DENSE_NORMAL_CHOLESKY、DENSE_SCHUR、DENSE_SVD等方法可以选择
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;    // 输出到cout

    Solver::Summary summary;    // 输出优化信息
    Solve(options, &problem, &summary);    // 进行求解

    cout<<summary.BriefReport()<<"\n";    // 输出优化的简要信息
    cout<<summary.FullReport()<<"\n";     // 输出优化的详细信息

    cout<<"初始值x= "<<initial_x<<"\n"<<"优化后值x= "<<x<<"\n";

    return 0;
}