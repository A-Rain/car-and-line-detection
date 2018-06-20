cmake_minimum_required(VERSION 3.10)
project(cv)

#include_directories(/usr/local/include/opencv)
#include_directories(/usr/local/include/opencv2)
set(CMAKE_CXX_STANDARD 14)
set(INC_DIR ./tinyxml)
set(LIB_DIR ./tinyxml)
include_directories(${INC_DIR})
link_directories(${LIB_DIR})
link_libraries(tinyxml)

find_package( OpenCV REQUIRED)
add_executable(cv main.cpp GetFeature.h GetFeature.cpp detection.h detection.cpp train.cpp train.h Config.h Config.cpp)
#set(TARGET_LIBS
#        "/usr/local/lib/libopencv_contrib.so"
#        "/usr/local/lib/libopencv_core.so"
#        "/usr/local/lib/libopencv_highgui.so"
#        "/usr/local/lib/libopencv_imgproc.so"
#        )

target_link_libraries(cv ${OpenCV_LIBS})
target_link_libraries(cv tinyxml)

