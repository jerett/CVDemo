

add_executable(eigenMatrix eigen_matrix.cc)
add_executable(useGeometry use_geometry.cc)

find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})
