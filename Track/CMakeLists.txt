
FIND_PACKAGE(OpenCV REQUIRED)
# find_package(Boost COMPONENTS system filesystem REQUIRED)


set(STAPLE_SRC
        staple/psr.h
        staple/fhog.cpp staple/fhog.h
        staple/staple_tracker.cpp staple/staple_tracker.h
        staple/sse.hpp )


set(OP_SRC
    op/Correlation.cc
)

set(TRACK_SRC 
    ${STAPLE_SRC}
    ${OP_SRC}
    main.cc
    MosseTracker.cc MosseTracker.h
    tracker.cc tracker.h  track_input.cc track_input.h
    )

source_group(TREE ${CMAKE_SOURCE_DIR}/Track FILES ${TRACK_SRC})

add_executable(Track ${TRACK_SRC})
target_include_directories(Track PRIVATE ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(Track PRIVATE ${OpenCV_LIBS} Util)

add_executable(TestTrackOP TestOP.cc)
target_include_directories(TestTrackOP PRIVATE ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(TestTrackOP PRIVATE ${OpenCV_LIBS} Util)

# configure_file(./goturn-files-master/goturn.prototxt . COPYONLY)
# configure_file(./goturn-files-master/goturn.caffemodel . COPYONLY)
