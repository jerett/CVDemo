
FIND_PACKAGE(OpenCV REQUIRED)
add_executable(mbb mbb.cc)
target_include_directories(mbb PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(mbb PRIVATE ${OpenCV_LIBS})


configure_file(../Resource/apple.jpg . COPYONLY)
configure_file(../Resource/orange.jpg . COPYONLY)