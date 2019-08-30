#include <iostream>
#include <opencv2/core/core.hpp>    // 使用其中的随机数函数
#include <ceres/ceres.h>
#include <chrono>

using namespace std;


// 第一部分：使用仿函数（functor）构建代价函数的计算模型
struct CURVE_FITTING_COST
{
    CURVE_FITTING_COST ( double x, double y ) : _x ( x ), _y ( y ) {}    // CURVE_FITTING_COST结构体的声明
    // 残差的计算
    template <typename T>
    bool operator() (
                     const T* const abc,     // 模型参数，有3维
                     T* residual ) const     // 残差
    {
        residual[0] = T ( _y ) - ceres::exp ( abc[0]*T (_x) *T (_x) + abc[1]*T (_x) + abc[2]); // y-exp(ax^2+bx+c)，即残差
        return true;
    }
    const double _x, _y;    // x,y数据
};


int main ( int argc, char** argv )
{
    double a=1.0, b=2.0, c=1.0;         // 真实参数值

    /* N=50  -> estimated a,b,c = 0.346821 2.51583 0.902763
     * N=100 -> estimated a,b,c = 0.891943 2.17039 0.944142
     * N=120 -> estimated a,b,c = 0.9252 2.1134 0.964529
     * N=150 -> estimated a,b,c = 1.01385 1.96533 1.02071
     * N=180 -> estimated a,b,c = 0.997486 2.00792 0.993689
     * N=200 -> estimated a,b,c = 0.997961 2.0075 0.99316
     * N=250 -> estimated a,b,c = 1.00006 1.9997 1.00035
     * ... ...
     */

    int N=100;                          // 产生若干个数据点
    double w_sigma=1.0;                 // 噪声Sigma（标准差）值（零均值高斯噪声）
    cv::RNG rng;                        // OpenCV随机数产生器
    double abc[3] = {0,0,0};            // abc参数的初始估计值

    vector<double> x_data, y_data;      // 数据

    cout<<"generating data: "<<endl;

    // x从0到1迭代，均匀间隔，100次；输出相应的x，y值
    for ( int i=0; i<N; i++ )
    {
        double x = i/100.0;
        x_data.push_back ( x );

        // rng.gaussian ( w_sigma )产生零均值高斯噪声
        y_data.push_back (exp ( a*x*x + b*x + c ) + rng.gaussian ( w_sigma ));
        cout<<x_data[i]<<" "<<y_data[i]<<endl;
    }


    // 第二部分：构建最小二乘问题
    ceres::Problem problem;
    for ( int i=0; i<N; i++ )
    {

        problem.AddResidualBlock
        (
            // 构建代价方程（CostFunction）：使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3> (new CURVE_FITTING_COST( x_data[i], y_data[i] )),
            nullptr,            // 核函数，这里不使用，为空
            abc                 // 待估计参数
        );
    }


    // 第三部分：配置求解器
    ceres::Solver::Options options;     // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_QR;  // 增量方程求解方法
    options.minimizer_progress_to_stdout = true;   // 输出到cout

    ceres::Solver::Summary summary;                // 优化信息
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve ( options, &problem, &summary );  // 开始优化
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;

    // 输出结果
    cout<<summary.FullReport() <<endl;
    cout<<"estimated a,b,c = ";
    for ( auto a:abc ) cout<<a<<" ";
    cout<<endl;

    return 0;
}

