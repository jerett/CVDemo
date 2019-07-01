
FIND_PACKAGE(OpenCV REQUIRED)

add_executable(filter main.cc)

target_include_directories(filter PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(filter PRIVATE ${OpenCV_LIBS})
configure_file(../Resource/lena.png . COPYONLY)