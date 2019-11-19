
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Ceres REQUIRED)

add_executable(orb orb.cc)
target_include_directories(orb PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        )
target_link_libraries(orb PRIVATE
        ${OpenCV_LIBS}
        )
configure_file(resource/1.png . COPYONLY)
configure_file(resource/2.png . COPYONLY)


add_executable(poseEstimation2d2d pose_estimation_2d2d.cc)
target_include_directories(poseEstimation2d2d PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        )

target_link_libraries(poseEstimation2d2d PRIVATE
        ${OpenCV_LIBS}
        )

add_executable(triangulation
        triangulation.cc)
target_include_directories(triangulation PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        )

target_link_libraries(triangulation PRIVATE
        ${OpenCV_LIBS}
        )
