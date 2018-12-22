cmake_minimum_required(VERSION 3.10)
project(OpenCVDemo)

set(CMAKE_CXX_STANDARD 11)
add_subdirectory(Stitch)
# uncomment if you have GLFW, DevIL, GLUT
#add_subdirectory(SiftGPUStitch)

add_subdirectory(Track)