
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Ceres REQUIRED)

add_executable(ceresCurveFitting ceres_curve_fitting.cc)

target_include_directories(ceresCurveFitting PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS})
target_link_libraries(ceresCurveFitting PRIVATE
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
        )

