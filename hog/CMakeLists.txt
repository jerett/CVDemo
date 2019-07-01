
FIND_PACKAGE(OpenCV REQUIRED)
add_executable(hog main.cc)
target_include_directories(hog PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hog PRIVATE ${OpenCV_LIBS})

configure_file(../Resource/person.png . COPYONLY)
