

add_executable(eigenMatrix eigen_matrix.cc)
add_executable(useGeometry useGeometry.cc)

find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})
