
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
find_package(EIGEN3)
include_directories(${EIGEN3_INCLUDE_DIR})

#configure_file(resource/1.png . COPYONLY)
#configure_file(resource/1_depth.png . COPYONLY)
#configure_file(resource/2.png . COPYONLY)
#configure_file(resource/2_depth.png . COPYONLY)
configure_file(resource/LK1.png . COPYONLY)
configure_file(resource/LK2.png . COPYONLY)


add_executable(opticalFlow optical_flow.cc)
target_link_libraries(opticalFlow PRIVATE
        ${OpenCV_LIBS}
        )

