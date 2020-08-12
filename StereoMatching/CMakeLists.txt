
FIND_PACKAGE(OpenCV REQUIRED)
add_executable(StereoMatching StereoMatching.cc)

set(CMAKE_CXX_STANDARD 11)

target_include_directories(StereoMatching PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(StereoMatching PRIVATE ${OpenCV_LIBS})


configure_file(../Resource/flow_1.jpg . COPYONLY)
configure_file(../Resource/flow_2.jpg . COPYONLY)
