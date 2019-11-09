//
// Created by jerett on 2019/11/9.
//

#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>

struct CurveFittingCost {
    CurveFittingCost(double x, double y) : x_(x), y_(y) {}

    template<typename T>
    bool operator()(const T *const abc, // 模型参数，有3维
                    T *residual) const {
        residual[0] = T(y_) - ceres::exp(abc[0] * T(x_) * T(x_) + abc[1] * T(x_) + abc[2]);
        return true;
    }

    const double x_, y_;
};

int main(int argc, char *argv[]) {
    std::cout << "test" << std::endl;
    double ar = 1.0, br = 2.0, cr = 1.0;
    double ae = 2.0, be = -1.0, ce = 5.0;
    int N = 100;
    double w_sigma = 1.0;
    cv::RNG rng;

    std::vector<double> x_data, y_data;
    for (int i = 0; i < N; ++i) {
        double x = i / 100;
        double y = exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma);
        x_data.push_back(x);
        y_data.push_back(y);
    }

    double abc[3] = {ae, be, ce};

    ceres::Problem problem;
    for (int i = 0; i < N; ++i) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CurveFittingCost, 1, 3>(new CurveFittingCost(x_data[i], y_data[i])),
            nullptr,
            abc
            );
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    auto t1 = std::chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    auto t2 = std::chrono::steady_clock::now();

    auto time = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
    std::cout << "solve time cost =" << time.count() << " seconds" << std::endl;

    std::cout << summary.BriefReport() << std::endl;
    std::cout << "estimate a,b,c=";
    for (auto p:abc) std::cout << p << " ";
    std::cout << std::endl;

    return 0;
}
