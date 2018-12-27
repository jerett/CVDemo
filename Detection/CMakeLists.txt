
FIND_PACKAGE(OpenCV REQUIRED)

add_executable(detection main.cc yolov3_detector.cc yolov3_detector.h)

target_include_directories(detection PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(detection PRIVATE ${OpenCV_LIBS})

#configure_file(models/coco.names . COPYONLY)
#configure_file(models/yolov3.cfg . COPYONLY)
#configure_file(models/yolov3.weights . COPYONLY)

configure_file(test_img/horses.jpg . COPYONLY)
configure_file(test_img/person.jpg . COPYONLY)
configure_file(test_img/messi.jpg . COPYONLY)
