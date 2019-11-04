//
// Created by jerett on 2019/10/25.
//

#include <iostream>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>

#define MATRIX_SIZE 50

using namespace std;

int main(int argc, char *argv[]) {
    Eigen::Matrix<float, 2, 3> matrix_23;
    Eigen::Vector3d v_3d;
    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic;
    Eigen::MatrixXd matrix_x;

    matrix_23 << 1, 2, 3, 4, 5, 6;
    std::cout << matrix_23 << std::endl;
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << matrix_23(i, j) << std::endl;
        }
    }
    v_3d << 3, 2, 1;
    std::cout << v_3d << std::endl;
    Eigen::Matrix<double, 2, 1> result_wrong_type = matrix_23.cast<double>() * v_3d;
    std::cout << result_wrong_type << std::endl;

    matrix_33 = Eigen::Matrix3d::Random();
    std::cout << "#######matrix33######" << std::endl;
    std::cout << matrix_33 << std::endl;
    std::cout << "#######matrix33 transpose######" << std::endl;
    std::cout << matrix_33.transpose() << std::endl;
    std::cout << "#######matrix33 sum######" << std::endl;
    std::cout << matrix_33.sum() << std::endl;
    std::cout << "#######matrix33 10x######" << std::endl;
    std::cout << matrix_33 * 10 << std::endl;
    std::cout << "#######matrix33 inverse######" << std::endl;
    std::cout << matrix_33.inverse() << std::endl;

    //
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_33);
    std::cout << "Eigen values = \n" << eigen_solver.eigenvalues() << std::endl;
    std::cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << std::endl;

    // matrix_NN * x = v_Nd
    Eigen::Matrix< double, MATRIX_SIZE, MATRIX_SIZE > matrix_NN;
    matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    Eigen::Matrix<double, MATRIX_SIZE, 1> v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);

    clock_t time = clock();
    Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
    cout <<"time use in normal invers is " << 1000* (clock() - time)/(double)CLOCKS_PER_SEC << "ms"
         << endl;

    time = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout <<"time use in Qr compsition is " << 1000* (clock() - time)/(double)CLOCKS_PER_SEC << "ms"
         << endl;
    return 0;
}