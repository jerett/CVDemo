
FIND_PACKAGE(OpenCV REQUIRED)
add_executable(Stitch stitch.cc)
target_include_directories(Stitch PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(Stitch PRIVATE ${OpenCV_LIBS})

configure_file(../Resource/uttower1.jpg . COPYONLY)
configure_file(../Resource/uttower2.jpg . COPYONLY)
configure_file(../Resource/Image1.jpg . COPYONLY)
configure_file(../Resource/Image2.jpg . COPYONLY)
configure_file(../Resource/yosemite1.jpg . COPYONLY)
configure_file(../Resource/yosemite2.jpg . COPYONLY)
configure_file(../Resource/yosemite3.jpg . COPYONLY)
configure_file(../Resource/yosemite4.jpg . COPYONLY)
