
FIND_PACKAGE(OpenCV REQUIRED)

add_executable(undistort undistort.cc)
target_include_directories(undistort PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(undistort PRIVATE ${OpenCV_LIBS})
configure_file(resource/distorted.png . COPYONLY)


find_package(Eigen3)
find_package(Pangolin REQUIRED)

add_executable(StereoVision stereo_vision.cc)
target_include_directories(StereoVision PRIVATE ${OpenCV_INCLUDE_DIRS} ${Pangolin_LIBRARIES})
target_link_libraries(StereoVision PRIVATE ${OpenCV_LIBS}  ${Pangolin_LIBRARIES})
target_include_directories(StereoVision PUBLIC ${EIGEN3_INCLUDE_DIR})
configure_file(resource/left.png . COPYONLY)
configure_file(resource/right.png . COPYONLY)


add_executable(jointMap jointMap.cc)
target_include_directories(jointMap PRIVATE ${OpenCV_INCLUDE_DIRS} ${Pangolin_LIBRARIES})
target_link_libraries(jointMap PRIVATE ${OpenCV_LIBS}  ${Pangolin_LIBRARIES})
target_include_directories(jointMap PUBLIC ${EIGEN3_INCLUDE_DIR})
file(COPY resource/color DESTINATION .)
file(COPY resource/depth DESTINATION .)
#configure_file(resource/color . COPYONLY)
#configure_file(resource/depth . COPYONLY)
configure_file(resource/pose.txt . COPYONLY)
