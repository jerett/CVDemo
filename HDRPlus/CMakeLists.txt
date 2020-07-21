

FIND_PACKAGE(OpenCV REQUIRED)
add_subdirectory(AdobeXMP)
add_subdirectory(dngsdk)
add_executable(HDRPlus 
        main.cc  
        RedundantDXTDenoise.cpp RedundantDXTDenoise.h
        dct_simd.cpp transpose_simd.cpp utility.cpp walsh_hadamard_simd.cpp
        HDRPlus.cc HDRPlus.h
        ISP.cc ISP.h
        DNGFile.cc DNGFile.h)
target_include_directories(HDRPlus PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(HDRPlus PRIVATE ${OpenCV_LIBS} libdng Util)
