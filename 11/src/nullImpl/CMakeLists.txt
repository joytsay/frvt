cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_definitions("-DWITH_EXTENSIONS")
#For the include headers:
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include)
MESSAGE("[DEBUG] Include headers: CMAKE_CURRENT_SOURCE_DIR{${CMAKE_CURRENT_SOURCE_DIR}}")

file(GLOB face_detection_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/dlib/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/dlib/*.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/dlib/dnn/*.h
    # ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/dlib/dnn/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include/*.cpp
    GVimplfrvt11.cpp
)

# Configure output built shared libraries in top-level lib directory:
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)

#For the shared library:
set(lib_dir ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)
MESSAGE("[DEBUG] lib_dir{${lib_dir}}")
find_library(opencv_highgui NAMES opencv_highgui PATHS ${lib_dir} NO_DEFAULT_PATH)
find_library(opencv_imgcodecs NAMES opencv_imgcodecs PATHS ${lib_dir} NO_DEFAULT_PATH)
find_library(opencv_imgproc NAMES opencv_imgproc PATHS ${lib_dir} NO_DEFAULT_PATH)
find_library(dlib NAMES dlib PATHS ${lib_dir} NO_DEFAULT_PATH)
find_library(tf NAMES tensorflow PATHS ${lib_dir} NO_DEFAULT_PATH)
find_library(tf_framework NAMES tensorflow_framework PATHS ${lib_dir} NO_DEFAULT_PATH)
# libtensorflow_framework

MESSAGE("[DEBUG] Libraries: opencv_imgcodecs{${opencv_imgcodecs}} opencv_imgproc{${opencv_imgproc}} dlib{${dlib}} tf{${tf}} tf_framework{${tf_framework}}") 

# Build the shared libraries:
add_library (frvt_11_geo_000 SHARED ${face_detection_sources})
target_link_libraries(frvt_11_geo_000 ${dlib} ${tf} ${tf_framework} ${opencv_imgproc} ${opencv_imgcodecs})

