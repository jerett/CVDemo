
FIND_PACKAGE(OpenCV REQUIRED)

add_executable(opencv_draw main.cc)

target_include_directories(opencv_draw PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv_draw PRIVATE ${OpenCV_LIBS})

