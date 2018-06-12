ADD_SUBDIRECTORY(SiftGPU)

FIND_PACKAGE(OpenCV REQUIRED)
add_executable(SiftGPUStitch siftgpu_stitch.cc)
target_include_directories(SiftGPUStitch PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(SiftGPUStitch PRIVATE siftgpu ${OpenCV_LIBS})

configure_file(../Resource/uttower1.jpg . COPYONLY)
configure_file(../Resource/uttower2.jpg . COPYONLY)
