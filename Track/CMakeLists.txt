
FIND_PACKAGE(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)

add_executable(track track.cc)

target_include_directories(track PRIVATE ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_libraries(track PRIVATE ${OpenCV_LIBS} ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY})

configure_file(./goturn-files-master/goturn.prototxt . COPYONLY)
configure_file(./goturn-files-master/goturn.caffemodel . COPYONLY)
