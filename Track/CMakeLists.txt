
FIND_PACKAGE(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)


set(STAPLE_SRC staple/fhog.cpp staple/fhog.h
        staple/staple_tracker.cpp staple/staple_tracker.hpp
        staple/sse.hpp track_input.cc track_input.h)
add_executable(track main.cc ${STAPLE_SRC} tracker.cc tracker.h)

target_include_directories(track PRIVATE ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_libraries(track PRIVATE ${OpenCV_LIBS} ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY})

configure_file(./goturn-files-master/goturn.prototxt . COPYONLY)
configure_file(./goturn-files-master/goturn.caffemodel . COPYONLY)
