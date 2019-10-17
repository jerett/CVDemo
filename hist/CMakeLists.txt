
FIND_PACKAGE(OpenCV REQUIRED)
add_executable(histMatch hist_match.cc)
target_include_directories(histMatch PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(histMatch PRIVATE ${OpenCV_LIBS})

configure_file(../Resource/DSC02984.JPG . COPYONLY)
configure_file(../Resource/DSC02986.JPG . COPYONLY)