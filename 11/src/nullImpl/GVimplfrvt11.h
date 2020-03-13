/*
 * This software was developed at the National Institute of Standards and
 * Technology (NIST) by employees of the Federal Government in the course
 * of their official duties. Pursuant to title 17 Section 105 of the
 * United States Code, this software is not subject to copyright protection
 * and is in the public domain. NIST assumes no responsibility  whatsoever for
 * its use by other parties, and makes no guarantees, expressed or implied,
 * about its quality, reliability, or any other characteristic.
 */

#ifndef NULLIMPLFRVT11_H_
#define NULLIMPLFRVT11_H_

#include "frvt11.h"
#include <fstream>
#include <utility>
#include <vector>

#include "tensorflow/c/c_api.h"
// #include "tensorflow_mtcnn.hpp"
// #include "mtcnn.hpp"
// #include "comm_lib.hpp"
// #include "utils.hpp"
// #include <unistd.h>

#include <time.h>
#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <memory>
#include <string>
#include <string.h>
#include <sys/stat.h>
#include <string>
#include <mutex>
#include <iostream>
#include <ctime>    
#include <stdio.h>
// #include <inference_engine.hpp>

//  #include <samples/ocv_common.hpp>
//#include <samples/slog.hpp>

//#include "interactive_face_detection.hpp"
// #include "detectors.hpp"
// #include "face.hpp"
// #include "visualizer.hpp"
// #include <ie_iextension.h>
// #include <ext_list.hpp>
// #define TBB_PREVIEW_GLOBAL_CONTROL 1
// #include <tbb/global_control.h>
#include <opencv2/opencv.hpp>
#include "facedetectcnn.h"
// #include "opencv2/highgui.hpp"
// #include <opencv2/imgcodecs.hpp>


#include "tf_utils.hpp"
#include <scope_guard.hpp>
#include <iostream>
#include <vector>

#define DLIB_JPEG_SUPPORT
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/geometry/rectangle.h>
#include <dlib/pixel.h>
#include <dlib/geometry/vector.h>
#include <dlib/pixel.h>
#include <dlib/opencv/to_open_cv.h>
#include <dlib/matrix/matrix.h>
#include <dlib/image_transforms/interpolation.h>
#include <dlib/image_transforms.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/full_object_detection.h>
#include <dlib/image_transforms.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_saver/save_jpeg.h>
// #include <dlib/dnn.h>

#define FR_IMAGE_HEIGHT 224
#define FR_IMAGE_PADDING 25
#define FR_EMBEDDING_SIZE 512
#define FR_JITTER_COUNT 0
#define DETECT_BUFFER_SIZE 0x20000
// ----------------------------------------------------------------------------------------

// template <long num_filters, typename SUBNET> using con5d = dlib::con<num_filters,5,5,2,2,SUBNET>;
// template <long num_filters, typename SUBNET> using con5  = dlib::con<num_filters,5,5,1,1,SUBNET>;

// template <typename SUBNET> using downsampler  = dlib::relu<dlib::affine<con5d<32, dlib::relu<dlib::affine<con5d<32, dlib::relu<dlib::affine<con5d<16,SUBNET>>>>>>>>>;
// template <typename SUBNET> using rcon5  = dlib::relu<dlib::affine<con5<45,SUBNET>>>;

// using net_type = dlib::loss_mmod<dlib::con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------

int detectFailCount = 0;
int imgCount = 0;
std::mutex mtx;
std::mutex saveImgMtx;

/*
 * Declare the implementation class of the FRVT 1:1 Interface
 */
namespace FRVT_11 {
    class NullImplFRVT11 : public FRVT_11::Interface {
public:

    NullImplFRVT11();
    ~NullImplFRVT11() override;

    FRVT::ReturnStatus
    initialize(const std::string &configDir) override;

    FRVT::ReturnStatus
    createTemplate(
            const FRVT::Multiface &faces,
            FRVT::TemplateRole role,
            std::vector<uint8_t> &templ,
            std::vector<FRVT::EyePair> &eyeCoordinates) override;

    FRVT::ReturnStatus
    matchTemplates(
            const std::vector<uint8_t> &verifTemplate,
            const std::vector<uint8_t> &enrollTemplate,
            double &similarity) override;

    static std::shared_ptr<FRVT_11::Interface>
    getImplementation();

private:
    std::string configDir;
    static const int featureVectorSize{FR_EMBEDDING_SIZE};
    // Some other members
    //unsigned char* input_image = NULL;
    // std::string input_name;
    // std::string output_name;
    //tbb::global_control *tbbControl = NULL;
    // std::string deviceName;
    // FaceDetection *faceDetector = NULL;
    // FacialLandmarksDetection *facialLandmarksDetector = NULL;
    // // --------------------------- 1. Load inference engine instance -------------------------------------
    // InferenceEngine::Core ie;
    // bool bFaceDetectorIsLoaded;
    // bool bFaceLandmarkIsLoaded;
    // // -----------------------------------------------------------------------------------------------------
    // // InferenceEngine::ExecutableNetwork executable_network;
    // InferenceEngine::InferRequest infer_request;


    //====================For FD====================//
    dlib::frontal_face_detector face_input_detector;
    dlib::shape_predictor sp_5;
    TF_Graph *graphFD;
    TF_Session *sessionFD;
    // net_type net;
    //===============================================//

    //====================For FR====================//
    TF_Graph *graph;
    // InferenceEngine::Core engine_ptr;
	// InferenceEngine::InferRequest infer_request;
    // InferenceEngine::CNNNetwork network;
	// std::string network_input_name;
    // static InferenceEngine::ExecutableNetwork::Ptr exe_network;
	std::vector<std::string> network_OutputName;
	std::map<std::string, int> OutputName_vs_index;
	unsigned char* input_image = NULL;
	// int batch_size;
	double mean_values[3];
	double scale_values[3];
    float FR_emb[512];
    dlib::matrix<float, 0, 1> test_matrix1;
    dlib::matrix<float, 0, 1> test_matrix2;
    int enrollCount;
    int faceDetectCount;
    // float gender[2];
    // float age[7];
    // int detectFailCount;
    // int imgCount;
	// std::vector<std::string> Device_List;
	// std::string Plugin_Device;
    int m_JitterCount;
    float jitterFR_emb[512];
    std::vector<dlib::matrix<dlib::rgb_pixel>> jitter_image(const dlib::matrix<dlib::rgb_pixel>& img, int height, int width);
    std::vector <dlib::matrix<dlib::rgb_pixel>> crops;
    std::vector<dlib::matrix<float, 0, 1>> array_to_dlib_1D_matrix(int face_count, float* in_array, int dim_size);
    std::string ProduceUUID();
    static void Deallocator(void* data, size_t length, void* arg);
    //===============================================//
};
}

#endif /* NULLIMPLFRVT11_H_ */
