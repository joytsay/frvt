cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_definitions("-DWITH_EXTENSIONS")
#For the include headers:
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/interactive_face_detection_demo ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/extension ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/extension/common)
MESSAGE("[DEBUG] Include headers: CMAKE_CURRENT_SOURCE_DIR{${CMAKE_CURRENT_SOURCE_DIR}}")

file(GLOB face_detection_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/interactive_face_detection_demo/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/interactive_face_detection_demo/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/extension/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/extension/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/extension/common/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/extension/common/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/extension/common/src/include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/extension/common/src/include/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/cpp/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/cpp/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/dlib/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/dlib/*.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/dlib/image_processing/*.h
    #${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/dlib/image_processing/*.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/dlib/image_transforms/*.h
    #${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/dlib/image_transforms/*.cpp
    GVimplfrvt11.cpp
)

# Configure output built shared libraries in top-level lib directory:
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)

#For the shared library:
#set(lib_dir ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/lib)
set(lib_dir ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)
MESSAGE("[DEBUG] lib_dir{${lib_dir}}")
find_library(cpu_extension NAMES cpu_extension PATHS ${lib_dir} NO_DEFAULT_PATH)
find_library(ie_inference_engine NAMES inference_engine PATHS ${lib_dir} NO_DEFAULT_PATH)
find_library(opencv_highgui NAMES opencv_highgui PATHS ${lib_dir} NO_DEFAULT_PATH)
find_library(ie_tbb NAMES tbb PATHS ${lib_dir} NO_DEFAULT_PATH)
find_library(opencv_core NAMES opencv_core PATHS ${lib_dir} NO_DEFAULT_PATH)
find_library(opencv_imgcodecs NAMES opencv_imgcodecs PATHS ${lib_dir} NO_DEFAULT_PATH)
find_library(opencv_imgproc NAMES opencv_imgproc PATHS ${lib_dir} NO_DEFAULT_PATH)
#find_library(opencv_videoio NAMES opencv_video PATHS ${lib_dir} NO_DEFAULT_PATH)
find_library(dlib NAMES dlib PATHS ${lib_dir} NO_DEFAULT_PATH)
MESSAGE("[DEBUG] Libraries: cpu_extension{${cpu_extension}} ie_inference_engine{${ie_inference_engine}} opencv_highgui{${opencv_highgui}} ie_tbb{${ie_tbb}} opencv_core{${opencv_core}} opencv_imgcodecs{${opencv_imgcodecs}} opencv_imgproc{${opencv_imgproc}} dlib{${dlib}}") #opencv_videoio{${opencv_videoio}} 

# Build the shared libraries:
add_library (frvt_11_geo_000 SHARED ${face_detection_sources})
target_link_libraries(frvt_11_geo_000 ${cpu_extension} ${ie_inference_engine} ${opencv_highgui} ${ie_tbb} ${opencv_core} ${opencv_imgcodecs} ${opencv_imgproc} ${opencv_videoio})

