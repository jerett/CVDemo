
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
find_package(G2O REQUIRED)

find_package(EIGEN3)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

configure_file(resource/1.png . COPYONLY)
configure_file(resource/1_depth.png . COPYONLY)
configure_file(resource/2.png . COPYONLY)
configure_file(resource/2_depth.png . COPYONLY)


add_executable(orb orb.cc)
target_link_libraries(orb PRIVATE
        ${OpenCV_LIBS}
        )

# poseEstimation2d2d
add_executable(poseEstimation2d2d pose_estimation_2d2d.cc)
target_link_libraries(poseEstimation2d2d PRIVATE
        ${OpenCV_LIBS}
        )

# triangulation
add_executable(triangulation triangulation.cc)
target_link_libraries(triangulation PRIVATE
        ${OpenCV_LIBS}
        )

# poseEstimation3d2d
add_executable(poseEstimation3d2d pose_estimation_3d2d.cc)
target_include_directories(poseEstimation3d2d PRIVATE
        ${G2O_INCLUDE_DIR}
        )
target_link_libraries(poseEstimation3d2d PRIVATE
        ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY}
        ${OpenCV_LIBS}
        )

# poseEstimation3d3d
add_executable(poseEstimation3d3d pose_estimation_3d3d.cc)
target_include_directories(poseEstimation3d3d PRIVATE
        ${G2O_INCLUDE_DIR}
        )
target_link_libraries(poseEstimation3d3d PRIVATE
        ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY}
        ${OpenCV_LIBS}
        )


