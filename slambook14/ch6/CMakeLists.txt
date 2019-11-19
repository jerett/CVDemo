
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message(${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Ceres REQUIRED)
find_package(G2O REQUIRED)


add_executable(ceresCurveFitting ceres_curve_fitting.cc)

target_include_directories(ceresCurveFitting PRIVATE
        ${CERES_INCLUDE_DIRS})

target_link_libraries(ceresCurveFitting PRIVATE
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
        )

add_executable(g2oCurveFitting g2o_curve_fitting.cc)
target_include_directories(g2oCurveFitting PRIVATE
        ${G2O_INCLUDE_DIR}
        )

target_link_libraries(g2oCurveFitting PRIVATE
        ${OpenCV_LIBS}
        ${G2O_CORE_LIBRARY}
        ${G2O_STUFF_LIBRARY}
        )
