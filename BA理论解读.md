## BA理论解读

### 1.  基本原理

#### 1.1 BA介绍

最小化重投影误差，获得最优的机器人位姿估计。bundle指的是光束，就是通过针孔相机模型获得的像素。重投影误差指的真实三维空间点在图像平面上的投影像素（真实值）和通过针孔相机模型计算得到的像素（估计值）差值。

> 这些东西归根结底就是Gauss“发明”的least squares method（最小二乘法）。当年天文学家Piazzi整天闲得没事看星星，在1801年1月1号早上发现了一个从来没观测到的星星，再接下来的42天里做了19次观测之后这个星星就消失了。当时的天文学家为了确定这玩意到底是什么绞尽了脑汁，这时候Gauss出现了，（最初）只用了3个观察数据，就用least squares算出了这个小行星的轨道，接下来天文学家根据Gauss的预测，也重新发现了这个小行星（虽然有小小的偏差），并将其命名为Ceres，也就是谷神星。Google的ceres-solver就是根据这个来命名的。

#### 1.2 BA建模

相机投影模型如下，其中：$s_i$为比例参数；$K$为内参矩阵（形式如式2）；$\exp \left(\boldsymbol{\xi}^{\wedge}\right)$为李代数表示的外参，$\xi$是一个1×6的向量（为什么不直接用一个4×4的矩阵直接表示外参呢？）
$$
s_{i}\left[\begin{array}{c}{u_{i}} \\ {v_{i}} \\ {1}\end{array}\right]=\boldsymbol{K} \exp \left(\boldsymbol{\xi}^{\wedge}\right)\left[\begin{array}{c}{X_{i}} \\ {Y_{i}} \\ {Z_{i}} \\ {1}\end{array}\right]
$$

其中：
$$
\boldsymbol{K}=\left[\begin{array}{ccc}{f_{x}} & {0} & {c_{x}} \\ {0} & {f_{y}} & {c_{y}} \\ {0} & {0} & {1}\end{array}\right]
$$

令：
$$
\boldsymbol{P}^{\prime}=\left(\exp \left(\boldsymbol{\xi}^{\wedge}\right) \boldsymbol{P}\right)_{1 : 3}=\left[X^{\prime}, Y^{\prime}, Z^{\prime}\right]^{\mathrm{T}}
$$
则
$$
\left[\begin{array}{c}{s u} \\ {s v} \\ {s}\end{array}\right]=\left[\begin{array}{ccc}{f_{x}} & {0} & {c_{x}} \\ {0} & {f_{y}} & {c_{y}} \\ {0} & {0} & {1}\end{array}\right]\left[\begin{array}{c}{X^{\prime}} \\ {Y^{\prime}} \\ {Z^{\prime}}\end{array}\right]
$$
由第三行可得$s=Z^{'}​$,则重投影（估计的）像素坐标为
$$
u=f_{x} \frac{X^{\prime}}{Z^{\prime}}+c_{x}, \quad v=f_{y} \frac{Y^{\prime}}{Z^{\prime}}+c_{y}
$$
构建最小二乘问题，寻找最优相机位姿
$$
\xi^{*}=\arg \min _{\xi} \frac{1}{2} \sum_{i=1}^{n}\left\|\boldsymbol{u}_{i}-\frac{1}{s_{i}} \boldsymbol{K} \exp \left(\boldsymbol{\xi}^{\wedge}\right) \boldsymbol{P}_{i}\right\|_{2}^{2}
$$
使用李代数，我们构建了无约束的优化问题，很方便地通过高斯牛顿法、L-M方法等优化算法进行求解。不过，在使用高斯牛顿法和L-M方法之前，我们需要知道每个误差项关于优化变量的导数，也就是线性化：

令
$$
e(x)=\boldsymbol{u}_{i}-\frac{1}{s_{i}} \boldsymbol{K} \exp \left(\boldsymbol{\xi}^{\wedge}\right) \boldsymbol{P}_{i}
$$
则
$$
e(x+\Delta x) \approx e(x)+J \Delta x
$$
构建最小二乘，对方程进行求导，得到下式（即我们常说的正规方程）
$$
\boldsymbol{J}(\boldsymbol{x}) e(\boldsymbol{x})+\boldsymbol{J}(\boldsymbol{x}) \boldsymbol{J}^{\mathrm{T}}(\boldsymbol{x}) \Delta \boldsymbol{x}=\mathbf{0}
$$
化简：
$$
\underbrace{J(x) J^{\mathrm{T}}}_{H(x)}(x) \Delta x=\underbrace{-J(x) f(x)}_{g(x)}
$$
得到增量方程
$$
\boldsymbol{H} \Delta \boldsymbol{x}=\boldsymbol{g}
$$

```
给定初始值 x0。
对于第 k 次迭代，求出当前的雅可比矩阵 J(x_k) 和误差 f(x_k)。
求解增量方程： H∆x_k = g。
若 ∆x_k 足够小，则停止。否则，令 x_{k+1} = x_k + ∆x_k，返回第 2 步。
```

这里的 $J$ 的形式是值得讨论的，甚至可以说是关键所在。
$$
J=\frac{\partial e}{\partial \delta \xi}=\lim _{\delta \xi \rightarrow 0} \frac{e(\delta \xi \oplus \xi)}{\delta \xi}=\frac{\partial e}{\partial P^{\prime}} \frac{\partial P^{\prime}}{\partial \delta \xi}
$$
其中
$$
\frac{\partial e}{\partial \boldsymbol{P}^{\prime}}=-\left[\begin{array}{ccc}{\frac{\partial u}{\partial X^{\prime}}} & {\frac{\partial u}{\partial Y^{\prime}}} & {\frac{\partial u}{\partial Z^{\prime}}} \\ {\frac{\partial v}{\partial X^{\prime}}} & {\frac{\partial v}{\partial Y^{\prime}}} & {\frac{\partial v}{\partial Z^{\prime}}}\end{array}\right]=-\left[\begin{array}{ccc}{\frac{f_{x}}{Z^{\prime}}} & {0} & {-\frac{f_{x} X^{\prime}}{Z^{\prime 2}}} \\ {0} & {\frac{f_{y}}{Z^{\prime}}} & {-\frac{f_{y} Y^{\prime}}{Z^{\prime 2}}}\end{array}\right]
$$

$$
\frac{\partial(\boldsymbol{T} \boldsymbol{P})}{\partial \delta \boldsymbol{\xi}}=(\boldsymbol{T} \boldsymbol{P})^{\odot}=\left[\begin{array}{cc}{\boldsymbol{I}} & {-\boldsymbol{P}^{\prime \wedge}} \\ {\mathbf{0}^{\mathrm{T}}} & {\mathbf{0}^{\mathrm{T}}}\end{array}\right]
$$

上面两项相乘
$$
J_{\xi}=\frac{\partial e}{\partial \delta \xi}=-\left[\begin{array}{cccccc}{\frac{f_{x}}{Z^{\prime}}} & {0} & {-\frac{f_{x} X^{\prime}}{Z^{\prime 2}}} & {-\frac{f_{x} X^{\prime} Y^{\prime}}{Z^{\prime 2}}} & {f_{x}+\frac{f_{x} X^{2}}{Z^{\prime 2}}} & {-\frac{f_{x} Y^{\prime}}{Z^{\prime}}} \\ {0} & {\frac{f_{y}}{z^{\prime}}} & {-\frac{f_{y} Y^{\prime}}{2^{\prime 2}}} & {-f_{y}-\frac{f_{y} Y^{\prime 2}}{Z^{\prime 2}}} & {\frac{f_{y} X^{\prime} Y^{\prime}}{Z^{\prime 2}}} & {\frac{f_{y} X^{\prime}}{Z^{\prime}}}\end{array}\right]
$$
另一方面，除了优化位姿，我们还希望优化特征点的空间位置。因此，需要讨论 e 关于空间点
P 的导数。所幸这个导数矩阵相对来说容易一些。仍利用链式法则，有：
$$
J_P=\frac{\partial e}{\partial P}=\frac{\partial e}{\partial P^{\prime}} \frac{\partial P^{\prime}}{\partial P}
$$

$$
J_P=\frac{\partial e}{\partial \boldsymbol{P}}=-\left[\begin{array}{ccc}{\frac{f_{x}}{Z^{\prime}}} & {0} & {-\frac{f_{x} X^{\prime}}{Z^{\prime 2}}} \\ {0} & {\frac{f_{y}}{Z^{\prime}}} & {-\frac{f_{y} Y^{\prime}}{Z^{\prime 2}}}\end{array}\right] \boldsymbol{R}
$$
上述方法适用于前端小型BA的实时求解。

对于大型BA的求解就不得不了解$H$矩阵的稀疏性。

![1567691035382](/home/sqg/.config/Typora/typora-user-images/1567691035382.png)

![1567691092817](/home/sqg/.config/Typora/typora-user-images/1567691092817.png)

![1567691123720](/home/sqg/.config/Typora/typora-user-images/1567691123720.png)

![1567691140838](/home/sqg/.config/Typora/typora-user-images/1567691140838.png)

### 2. 非线性最小二乘与因子图之间的联系

Dellaert, F.和 Kaess, M的论文Square Root SAM中揭示了因子图与非线性最小二乘之间紧密的联系。因子图是一个概率图形模型，它表示所有因子的联合概率分布
$$
p(x) \propto \prod_i p_i(x_i)
$$
其中$x_i \subseteq x$是涉及因子 𝑖 子集的变量，$p(x)$是因子图的整体分布，$p_i(x_i)​$ 是每个因子的分布。该图的最大后验（MAP）估计为
$$
x^{*} = \operatorname*{argmax}_{x} p(x) = \operatorname*{argmax}_{x} \prod_i p_i(x_i).
$$
如果考虑每个因子在$f_i(x_i)$上具有高斯分布且协方差$Σ_i$的情况，
$$
p_i(x_i) \propto \mathrm{exp} \big( - \frac{1}{2} \parallel f_i(x_i) \parallel^{2}_{{\Sigma}_i} \big),
$$
那么MAP推断是
$$
\begin{split}x^{*} & = \operatorname*{argmax}_{x} \prod_i p_i(x_i) = \operatorname*{argmax}_{x} \mathrm{log} \big( \prod_i p_i(x_i) \big),  \\
& = \operatorname*{argmin}_{x} \prod_i -\mathrm{log} \big( p_i(x_i) \big) = \operatorname*{argmin}_{x} \sum_i \parallel f_i(x_i) \parallel^{2}_{{\Sigma}_i}.\end{split}
$$
等式中的MAP推理问题被转换成上面提到的形式相同的非线性最小二乘优化问题。可以按照前一节中相同的步骤解决。

使用因子图对SLAM中的非线性最小二乘问题建模有几个优点。因子图对问题的概率性质进行编码，并且可以轻松地可视化大多数SLAM问题的潜在稀疏性，因为大多数（如果不是全部）因子$x_i$都是很小的集合。

### 3. 代码实践（使用ceres）

[参考](http://grail.cs.washington.edu/projects/bal/)

**假设：**使用针孔相机模型；相机旋转$R$，平移$t$，焦距$f$和两个径向畸变参数$k1$和$k2$。将3D点$X$投影到摄像机为参数$R，t，f，k1，k2​$表示u的公式为：

```
P = R * X + t（从世界坐标转换为相机坐标）

p = -P / P.z（归一化除法）

p'= f * r（p）* p（转换为像素坐标）
```

其中P.z是P的第三个（z）坐标。在最后一个方程中，r（p）是一个消除径向畸变的函数：

```
r(p) = 1.0 + k1 * ||p||^2 + k2 * ||p||^4.
```

这样就给出了以像素为单位的投影，其中图像的原点是图像的中心，x轴的正指向右，y轴的正指向上方（此外，在相机坐标系中，z轴为正轴指向后方，因此相机向下看Z轴的负方向。

```C++
#include <cmath>
#include <cstdio>
#include <iostream>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

// 读取 Bundle Adjustment in the Large 数据集.
class BALProblem {
 public:
  ~BALProblem()    // 析构函数：自动/手动释放对象使用的资源，销毁非static成员
  {
    delete[] point_index_;
    delete[] camera_index_;
    delete[] observations_;
    delete[] parameters_;
  }

  // 一些类内函数
  int num_observations()       const { return num_observations_;               }
  const double* observations() const { return observations_;                   }
  double* mutable_cameras()          { return parameters_;                     }
  double* mutable_points()           { return parameters_  + 9 * num_cameras_; }

  double* mutable_camera_for_observation(int i)
  {
    return mutable_cameras() + camera_index_[i] * 9;
  }

  double* mutable_point_for_observation(int i) {
    return mutable_points() + point_index_[i] * 3;
  }

  // 读取文件
  bool LoadFile(const char* filename) {
    FILE* fptr = fopen(filename, "r");
    if (fptr == nullptr) {
      return false;
    };

    FscanfOrDie(fptr, "%d", &num_cameras_);
    FscanfOrDie(fptr, "%d", &num_points_);
    FscanfOrDie(fptr, "%d", &num_observations_);

    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];

    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];

    for (int i = 0; i < num_observations_; ++i) {
      FscanfOrDie(fptr, "%d", camera_index_ + i);
      FscanfOrDie(fptr, "%d", point_index_ + i);
      for (int j = 0; j < 2; ++j) {
        FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
      }
    }

    for (int i = 0; i < num_parameters_; ++i) {
      FscanfOrDie(fptr, "%lf", parameters_ + i);
    }
    return true;
  }

 private:

  // 模板函数，读取文件或中断
  template<typename T>
  void FscanfOrDie(FILE *fptr, const char *format, T *value) {
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1) {
      LOG(FATAL) << "Invalid UW data file.";
    }
  }

  // 类内初始化
  int num_cameras_;
  int num_points_;
  int num_observations_;
  int num_parameters_;

  int* point_index_;
  int* camera_index_;
  double* observations_;
  double* parameters_;
};

// 计算重投影误差
// 构建针孔相机模型。使用9个参数对摄像机进行参数设置：3个用于旋转，3个用于平移，
// 1个用于焦距和2个用于径向畸变（假定主点位于图像中心）。
struct SnavelyReprojectionError {
  // 初始化构造函数
  SnavelyReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
    // camera[0] camera[1] camera[2] 为angle-axis旋转.
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);

    // camera[3] camera[4] camera[5] 为平移矩阵.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // 计算畸变的中心。符号变化来自Snavely的Bundler中所采用的相机模型，因此相机坐标系的Z轴为负。
    T xp = - p[0] / p[2];
    T yp = - p[1] / p[2];

    // 使用第二项和第四项径向畸变，即camera[7] camera[8]。
    const T& l1 = camera[7];
    const T& l2 = camera[8];
    T r2 = xp*xp + yp*yp;
    T distortion = 1.0 + r2  * (l1 + l2  * r2);

    // Compute final projected point position.
    // 计算最终的投影点位置，其中camera[6]为焦距参数。
    const T& focal = camera[6];
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp;

    // 误差是预测位置和观察位置之间的差异，求x和y的残差。
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;

    return true;
  }

  // 将CostFunction对象的构造隐藏在类内。
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
    // 使用自动求导
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                new SnavelyReprojectionError(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  if (argc != 2) {
    std::cerr << "usage: simple_bundle_adjuster <bal_problem>\n";
    return 1;
  }

  BALProblem bal_problem{};
  if (!bal_problem.LoadFile(argv[1])) {
    std::cerr << "ERROR: unable to open file " << argv[1] << "\n";
    return 1;
  }

  const double* observations = bal_problem.observations();

  // 为BA问题中的每个观测值创建残差。自动添加摄像机和点的参数。
  ceres::Problem problem;
  for (int i = 0; i < bal_problem.num_observations(); ++i) {
    // 每个残差块均以一个点和一个摄像机作为输入，并输出2维残差。
    // 在内部，cost function 存储观察到的图像位置，并将重投影与观察值进行比较。
    ceres::CostFunction* cost_function =
        SnavelyReprojectionError::Create(observations[2 * i + 0],
                                         observations[2 * i + 1]);
    problem.AddResidualBlock(cost_function,
                             nullptr /* squared loss */,
                             bal_problem.mutable_camera_for_observation(i),
                             bal_problem.mutable_point_for_observation(i));
  }

  // 使用Ceres自动求解。注意，标准求解器SPARSE_NORMAL_CHOLESKY
  // 也可以正常工作，但是对于标准BA问题来说速度较慢。（DENSE_SCHUR速度会比较快）
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
  return 0;
}

```



**数据集**：BAL数据集，来源于 [Building Rome in a Day](http://grail.cs.washington.edu/rome)项目。使用到的数据中包含16台相机拍摄到的22106特征点

```
数据集部分数据，其按下面格式排列
<num_cameras> <num_points> <num_observations>
<camera_index_1> <point_index_1> <x_1> <y_1>


16 22106 83718
0 0     -3.859900e+02 3.871200e+02
1 0     -3.844000e+01 4.921200e+02
2 0     -6.679200e+02 1.231100e+02
7 0     -5.991800e+02 4.079300e+02
12 0     -7.204300e+02 3.143400e+02
13 0     -1.151300e+02 5.548999e+01
0 1     3.838800e+02 -1.529999e+01
1 1     5.597500e+02 -1.061500e+02
10 1     3.531899e+02 1.649500e+02
0 2     5.915500e+02 1.364400e+02
1 2     8.638600e+02 -2.346997e+01
2 2     4.947200e+02 1.125200e+02
6 2     4.087800e+02 2.846700e+02
7 2     4.246100e+02 3.101700e+02
9 2     2.848900e+02 1.928900e+02
10 2     5.826200e+02 3.637200e+02
12 2     4.940601e+02 2.939500e+02
13 2     7.968300e+02 -7.853003e+01
15 2     7.798900e+02 4.082500e+02
```



使用DENSE_SCHUR方法求解

```
iter      cost      cost_change  |gradient|   |step|    tr_ratio  tr_radius  ls_iter  iter_time  total_time
   0  4.185660e+06    0.00e+00    1.09e+08   0.00e+00   0.00e+00  1.00e+04        0    7.20e+00    7.90e+00
   
......

   6  1.803390e+04    9.02e-02    6.35e+01   8.00e-01   1.00e+00  2.50e+06        1    3.28e+01    2.03e+02（约合3.4min）

                                     Original                  Reduced
Parameter blocks                        22122                    22122
Parameters                              66462                    66462
Residual blocks                         83718                    83718
Residuals                              167436                   167436

Minimizer                        TRUST_REGION

Dense linear algebra library            EIGEN
Trust region strategy     LEVENBERG_MARQUARDT

                                        Given                     Used
Linear solver                     DENSE_SCHUR              DENSE_SCHUR
Threads                                     1                        1
Linear solver ordering              AUTOMATIC                 22106,16
Schur structure                         2,3,9                    2,3,9

Cost:
Initial                          4.185660e+06
Final                            1.803390e+04
Change                           4.167626e+06

Minimizer iterations                        7
Successful steps                            7
Unsuccessful steps                          0

Time (in seconds):
Preprocessor                         0.700777

  Residual only evaluation           0.667742 (7)
  Jacobian & residual evaluation    39.360399 (7)
  Linear solver                    183.064203 (7)
Minimizer                          229.831484

Postprocessor                        0.014315
Total                              230.546577

Termination:                      CONVERGENCE (Function tolerance reached. |cost_change|/cost: 1.769759e-09 <= 1.000000e-06)

```



使用SPARSE_SCHUR求解

```
iter      cost      cost_change  |gradient|   |step|    tr_ratio  tr_radius  ls_iter  iter_time  total_time
   0  4.185660e+06    0.00e+00    1.09e+08   0.00e+00   0.00e+00  1.00e+04        0    6.34e+00    6.86e+00
   
......

   6  1.803390e+04    9.02e-02    6.35e+01   8.00e-01   1.00e+00  2.50e+06        1    3.30e+01    2.05e+02（约合3.4min）

                                     Original                  Reduced
Parameter blocks                        22122                    22122
Parameters                              66462                    66462
Residual blocks                         83718                    83718
Residuals                              167436                   167436

Minimizer                        TRUST_REGION

Sparse linear algebra library    SUITE_SPARSE
Trust region strategy     LEVENBERG_MARQUARDT

                                        Given                     Used
Linear solver                    SPARSE_SCHUR             SPARSE_SCHUR
Threads                                     1                        1
Linear solver ordering              AUTOMATIC                 22106,16
Schur structure                         2,3,9                    2,3,9

Cost:
Initial                          4.185660e+06
Final                            1.803390e+04
Change                           4.167626e+06

Minimizer iterations                        7
Successful steps                            7
Unsuccessful steps                          0

Time (in seconds):
Preprocessor                         0.524276

  Residual only evaluation           0.655267 (7)
  Jacobian & residual evaluation    38.889743 (7)
  Linear solver                    185.728749 (7)
Minimizer                          231.991205

Postprocessor                        0.010115
Total                              232.525597

Termination:                      CONVERGENCE (Function tolerance reached. |cost_change|/cost: 1.769757e-09 <= 1.000000e-06)
```



使用SPARSE_NORMAL_CHOLESKY方法求解

```
iter      cost      cost_change  |gradient|   |step|    tr_ratio  tr_radius  ls_iter  iter_time  total_time
   0  4.185660e+06    0.00e+00    1.09e+08   0.00e+00   0.00e+00  1.00e+04        0    6.46e+00    6.63e+00
   
......

   6  1.803390e+04    9.02e-02    6.35e+01   8.00e-01   1.00e+00  2.50e+06        1    6.63e+00    4.68e+01（约合0.8min）


                                     Original                  Reduced
Parameter blocks                        22122                    22122
Parameters                              66462                    66462
Residual blocks                         83718                    83718
Residuals                              167436                   167436

Minimizer                        TRUST_REGION

Sparse linear algebra library    SUITE_SPARSE
Trust region strategy     LEVENBERG_MARQUARDT

                                        Given                     Used
Linear solver          SPARSE_NORMAL_CHOLESKY   SPARSE_NORMAL_CHOLESKY
Threads                                     1                        1
Linear solver ordering              AUTOMATIC                    22122

Cost:
Initial                          4.185660e+06
Final                            1.803390e+04
Change                           4.167626e+06

Minimizer iterations                        7
Successful steps                            7
Unsuccessful steps                          0

Time (in seconds):
Preprocessor                         0.173184

  Residual only evaluation           0.591482 (7)
  Jacobian & residual evaluation    38.593874 (7)
  Linear solver                      1.753153 (7)
Minimizer                           47.594235

Postprocessor                        0.009093
Total                               47.776512

Termination:                      CONVERGENCE (Function tolerance reached. |cost_change|/cost: 1.769774e-09 <= 1.000000e-06)
```



使用DENSE_NORMAL_CHOLESKY求解

```C++
terminate called after throwing an instance of 'std::bad_alloc'
```

提示超内存了......





