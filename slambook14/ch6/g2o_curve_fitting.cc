//
// Created by jerett on 2019/11/19.
//

#include <iostream>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/linear_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

using namespace std;

class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // 重置
    virtual void setToOriginImpl() override {
        _estimate << 0, 0, 0;
    }

    // 更新
    virtual void oplusImpl(const double *update) override {
        std::cout << update[0] << " " << update[1] << " " << update[2] << std::endl;
        _estimate += Eigen::Vector3d(update);
    }

    // 存盘和读盘：留空
    virtual bool read(std::istream &in) override {return false;}

    virtual bool write(std::ostream &out) const override {return false;}

};

class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingEdge(double x) : BaseUnaryEdge(), x_(x) {}

    virtual void computeError() override {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
        const Eigen::Vector3d &abc = v->estimate();
        double y = std::exp(abc[0] * x_ * x_ + abc[1] * x_ + abc[2]);
        _error(0, 0) = _measurement - y;
    }

    virtual void linearizeOplus() override {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
        const Eigen::Vector3d &abc = v->estimate();
        double y = std::exp(abc[0] * x_ * x_ + abc[1] * x_ + abc[2]);
        _jacobianOplusXi[0] = -x_ * x_ * y;
        _jacobianOplusXi[1] = -x_ * y;
        _jacobianOplusXi[2] = -y;
    }

    // 存盘和读盘：留空
    virtual bool read(std::istream &in) override { return false;}

    virtual bool write(std::ostream &out) const override {return false;}

public:
    double x_; // y : _measurement
};

int main(int argc, char *argv[]) {
    std::cout << "g2o test" << std::endl;
    double ar = 1.0, br = 2.0, cr = 1.0;
    double ae = 2.0, be = -1.0, ce = 5.0;
    int N = 100;
    double w_sigma = 1.0;
    cv::RNG rng;

    std::vector<double> x_data, y_data;
    for (int i = 0; i < N; ++i) {
        double x = i / 100.0;
        double y = exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma);
        x_data.push_back(x);
        y_data.push_back(y);
    }

    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;
    auto linear_solver = g2o::make_unique<LinearSolverType>();
    auto block_solver = g2o::make_unique<BlockSolverType>(std::move(linear_solver));
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(std::move(block_solver));

    g2o::SparseOptimizer optim;
    optim.setAlgorithm(solver);
    optim.setVerbose(true);

    // add vertex
    CurveFittingVertex *v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(ae, be, ce));
    v->setId(0);
    optim.addVertex(v);

    // add edge
    for (int i = 0; i < N; ++i) {
        CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0, v);
        edge->setMeasurement(y_data[i]);
        // 信息矩阵：协方差矩阵之逆
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma));
        optim.addEdge(edge);
    }

    std::cout << "start opt." << std::endl;
    auto t1 = chrono::steady_clock::now();
    optim.initializeOptimization();
    optim.optimize(10);
    auto t2 = chrono::steady_clock::now();
    std::cout << "solve time :" << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() << std::endl;
    auto abc_estimate = v->estimate();
    std::cout << "estimated model:\n" << abc_estimate.transpose() << std::endl;

    return 0;
}
