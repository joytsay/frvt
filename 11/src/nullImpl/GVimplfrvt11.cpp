/*
 * This software was developed at the National Institute of Standards and
 * Technology (NIST) by employees of the Federal Government in the course
 * of their official duties. Pursuant to title 17 Section 105 of the
 * United States Code, this software is not subject to copyright protection
 * and is in the public domain. NIST assumes no responsibility  whatsoever for
 * its use by other parties, and makes no guarantees, expressed or implied,
 * about its quality, reliability, or any other characteristic.
 */

#include <cstring>
#include <cstdlib>

#include "GVimplfrvt11.h"

using namespace std;
using namespace FRVT;
using namespace FRVT_11;

InferenceEngine::ExecutableNetwork::Ptr NullImplFRVT11::exe_network = nullptr;

NullImplFRVT11::NullImplFRVT11() {}

NullImplFRVT11::~NullImplFRVT11() {
    // std::cout<<"~01"<<endl;
    if(faceDetector){
        // std::cout<<"~02"<<endl;
        delete faceDetector;
        faceDetector = NULL;
        // std::cout<<"~02-1"<<endl;
    }
        
    if(facialLandmarksDetector){
        // std::cout<<"~03"<<endl;
        delete facialLandmarksDetector;
        facialLandmarksDetector = NULL;
        // std::cout<<"~03-1"<<endl;
    }
       
    if(input_image){
        // std::cout<<"~04-1"<<endl;
        delete[] input_image;
        input_image = NULL;
        // std::cout<<"~04"<<endl;
    }
    // slog::info << "~NullImplFRVT11 imgCount: : " << imgCount << slog::endl;
}

ReturnStatus
NullImplFRVT11::initialize(const std::string &configDir)
{
	try { 
        detectFailCount = 0;
        if(!input_image){
            input_image = new unsigned char [FR_IMAGE_HEIGHT * FR_IMAGE_HEIGHT *3];
        }	
// std::cout<<"01"<<endl;
        // --------------------------- 1. Loading Inference Engine --------------------------------------------
        // ------------------------------ Parsing and validation of input args -----------------
        deviceName = "CPU";
        // slog::info << "configDir: " << configDir << " deviceName: " << deviceName 
                    // << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << slog::endl;
        std::string FDxmlFileName = configDir + "/yufacedetectnet-open-v1.xml";
        // slog::info << "FDxmlFileName: " << FDxmlFileName << slog::endl;
        std::string LMxmlFileName = configDir + "/facial-landmarks-35-adas-0001.xml";
        // slog::info << "LMxmlFileName: " << LMxmlFileName << slog::endl;
        std::string FRxmlFileName = configDir + "/#38_asian_multi_task_10.xml";
        // slog::info << "FRxmlFileName: " << FRxmlFileName << slog::endl;
		// -------------------------------------------------------------------------------------
// std::cout<<"02"<<endl;
        bool FLAGS_async = false;
        double FLAGS_t = 0.3;
        bool FLAGS_r = false;
        float FLAGS_bb_enlarge_coef = 1.2;
        float FLAGS_dx_coef = 1;
        float FLAGS_dy_coef = 1;
        faceDetector = new FaceDetection(FDxmlFileName, deviceName, 1, false, 
                            FLAGS_async, FLAGS_t, FLAGS_r, 
                            static_cast<float>(FLAGS_bb_enlarge_coef), 
                            static_cast<float>(FLAGS_dx_coef), 
                            static_cast<float>(FLAGS_dy_coef));
        int FLAGS_n_lm = 16;
        bool FLAGS_dyn_lm = false;
        facialLandmarksDetector = new FacialLandmarksDetection(LMxmlFileName, deviceName, 
                            FLAGS_n_lm, FLAGS_dyn_lm, FLAGS_async, FLAGS_r);
        // ----------------------------------------------------------------------------------------------------


        // --------------------------- 2. Reading IR models and loading them to plugins ----------------------
        // Disable dynamic batching for face detector as it processes one image at a time
        // slog::info << "Loading device " << deviceName << slog::endl;
        // std::cout << ie.GetVersions(deviceName) << std::endl;
        ie.AddExtension(std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions>(), deviceName);
        bFaceDetectorIsLoaded = false;
        bFaceLandmarkIsLoaded = false;

        // --------------------Load FR network (Generated xml/bin files)----------------------------------------
        /** Extracting model name and loading weights **/
        InferenceEngine::CNNNetReader networkReader;
        std::string model_path = FRxmlFileName;
        networkReader.ReadNetwork(model_path);
        std::string binFileName = model_path.substr(0, model_path.size() - 4) + ".bin";
        networkReader.ReadWeights(binFileName);
        network = networkReader.getNetwork();
        mean_values[0]  = mean_values[1]  = mean_values[2]  = 255.0*0.5;
	    scale_values[0] = scale_values[1] = scale_values[2] = 255.0*0.5;
        OutputName_vs_index = std::map<std::string, int >{	{ "normalize",0 },
                                                    { "Softmax_4",1 },
                                                    { "Softmax_5",2 } };
        // -----------------------------------------------------------------------------------------------------

        /** Creating FR input blob **/
        // -----------------------------Prepare input blobs-----------------------------------------------------
        /** Taking information about all topology inputs **/
        InferenceEngine::InputsDataMap inputInfo(network.getInputsInfo());
        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");
        auto inputInfoItem = *inputInfo.begin();
        network_input_name = inputInfoItem.first;
        inputInfoItem.second->setLayout(InferenceEngine::Layout::NCHW);
        network.setBatchSize(1);
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Prepare output blobs------------------------------------------------------
        InferenceEngine::OutputsDataMap outputInfo(network.getOutputsInfo());
        for (auto & item : outputInfo) {
            network_OutputName.push_back(item.first);
            InferenceEngine::DataPtr outputData = item.second;
            if (!outputData) {
                slog::err << "output data pointer is not valid" << slog::endl;
            }
            item.second->setLayout(InferenceEngine::Layout::NC);
        }
        // -----------------------------------------------------------------------------------------------------


        
        // -----------------------------------------------------------------------------------------------------
// std::cout<<"03"<<endl;
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
    }
	return ReturnStatus(ReturnCode::Success);
}

ReturnStatus
NullImplFRVT11::createTemplate(
        const Multiface &faces,
        TemplateRole role,
        std::vector<uint8_t> &templ,
        std::vector<EyePair> &eyeCoordinates)
{
    try { //---------------------------- Implement intel inference engine -------------------------------------
        // imgCount++;
        if(!bFaceDetectorIsLoaded){
            Load(*faceDetector).into(ie, deviceName, false);
            bFaceDetectorIsLoaded = true;
        }
        if(!bFaceLandmarkIsLoaded){
            Load(*facialLandmarksDetector).into(ie, deviceName, false);
            bFaceLandmarkIsLoaded = true;
        }

        // -------------------------Load FR model to the plugin-------------------------------------------------
        if (exe_network == nullptr){
            exe_network = std::make_shared<InferenceEngine::ExecutableNetwork>(ie.LoadNetwork(network, deviceName));
        }
        infer_request = exe_network->CreateInferRequest();

	    InferenceEngine::Blob::Ptr input = infer_request.GetBlob(network_input_name);
        InferenceEngine::SizeVector input_shape = input->getTensorDesc().getDims();
        size_t num_channels = input_shape[1];
        size_t image_size = input_shape[2] * input_shape[3];
        auto data = input->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        
        for (unsigned int i=0; i<faces.size(); i++) {
            // ----------------------------------------------------------------------------------------------------
// std::cout<<"04"<<endl;
            std::list<Face::Ptr> facesAttributes;
            size_t id = 0;
            cv::Mat frame = cv::Mat(faces[i].height, faces[i].width, CV_8UC3);
            // ------------------------------Visualizing results------------------------------------------------
            Visualizer::Ptr visualizer;
            visualizer = std::make_shared<Visualizer>(cv::Size(faces[i].width, faces[i].height ));
            // -------------------------------------------------------------------------------------------------
// std::cout<<"05"<<endl;
            // -------------------------------Set input data----------------------------------------------------
            slog::info << "frvt imput image height: " << faces[i].height << ", width: " << faces[i].width << ", size: " << faces[i].size() << slog::endl;
            std::memcpy(frame.data, faces[i].data.get(), faces[i].size() );  
            cv::cvtColor(frame,frame, cv::COLOR_BGR2RGB);
            cv::imshow("Origin image", frame);
            cv::waitKey(100);
            cv::destroyAllWindows(); 
            // Detecting all faces on the frame
            faceDetector->enqueue(frame);
// std::cout<<"06"<<endl;
        faceDetector->submitRequest();
// std::cout<<"07"<<endl;
        // Retrieving face detection results for the frame
        faceDetector->wait();
        faceDetector->fetchResults();
        auto prev_detection_results = faceDetector->results;
        slog::info << "prev_detection_results.size(): " << prev_detection_results.size() << slog::endl;
// std::cout<<"08"<<endl;
            // Filling inputs of face analytics networks
            for (auto &&face : prev_detection_results) {
                    auto clippedRect = face.location & cv::Rect(0, 0, frame.cols, frame.rows);
                    cv::Mat clippedFace = frame(clippedRect);
                    facialLandmarksDetector->enqueue(clippedFace);
            }
// std::cout<<"09"<<endl;
            // Running Facial Landmarks Estimation networks simultaneously
            facialLandmarksDetector->submitRequest();
// std::cout<<"10"<<endl;
            // For every detected face
            for (size_t j = 0; j < prev_detection_results.size(); j++) {
                auto& result = prev_detection_results[j];
                cv::Rect rect = result.location & cv::Rect(0, 0, frame.cols, frame.rows);
                Face::Ptr face;
                face = std::make_shared<Face>(id++, rect);
                face->landmarksEnable((facialLandmarksDetector->enabled() &&
                                    j < facialLandmarksDetector->maxBatch));
                if (face->isLandmarksEnabled()) {
                    face->updateLandmarks((*facialLandmarksDetector)[j]);
                }
                facesAttributes.push_back(face);
// std::cout<<"11"<<endl;
                auto normed_landmarks = (*facialLandmarksDetector)[j];
                auto n_lm = normed_landmarks.size();
                for (auto i_lm = 0UL; i_lm < n_lm / 2; ++i_lm) {
                    float normed_x = normed_landmarks[2 * i_lm];
                    float normed_y = normed_landmarks[2 * i_lm + 1];
                    int x_lm = rect.x + rect.width * normed_x;
                    int y_lm = rect.y + rect.height * normed_y;
                    // slog::info << "landmark("<< i_lm << "): (x,y)=(" << x_lm << "," << y_lm << ")"  << slog::endl;
                    // cv::circle(frame, cv::Point(x_lm, y_lm), 1 + static_cast<int>(0.012 * rect.width), cv::Scalar(0, 255, 255), -1);
                    // string lmText = to_string(i_lm); 
                    // cv::putText(frame, lmText, cv::Point(x_lm, y_lm), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0));
                }
                // --------------------------- Do Face and Landmark Detection for eye center----------------------------
                int xleftEyeCenter = int (0.5 * (rect.x + rect.width * (normed_landmarks[2 * 0]) + rect.x + rect.width * (normed_landmarks[2 * 1])));
                int yleftEyeCenter = int (0.5 * (rect.y + rect.height * (normed_landmarks[2 * 0 + 1]) + rect.y + rect.height * (normed_landmarks[2 * 1 + 1])));
                cv::circle(frame, cv::Point(xleftEyeCenter, yleftEyeCenter), 1 + static_cast<int>(0.012 * rect.width), cv::Scalar(255, 0, 0), -1);
                int xRightEyeCenter = int (0.5 * (rect.x + rect.width * (normed_landmarks[2 * 2]) + rect.x + rect.width * (normed_landmarks[2 * 3])));
                int yRightEyeCenter = int (0.5 * (rect.y + rect.height * (normed_landmarks[2 * 2 + 1]) + rect.y + rect.height * (normed_landmarks[2 * 3 + 1])));
                cv::circle(frame, cv::Point(xRightEyeCenter, yRightEyeCenter), 1 + static_cast<int>(0.012 * rect.width), cv::Scalar(0, 0, 255), -1);
                // eyeCoordinates.clear();
                // eyeCoordinates.shrink_to_fit();
                eyeCoordinates.push_back(EyePair(true, true, xRightEyeCenter, yRightEyeCenter, xleftEyeCenter, yleftEyeCenter));
                ////////////ISO standard: The label "left" refers to subject's left eye (and similarly for the right eye), such that xright < xleft/////////////////
                slog::info << "eyeCoordinatesLeftEye("<< i << "): (x,y)=(" << eyeCoordinates[i].xleft << "," << eyeCoordinates[i].yleft << ")"  << slog::endl;
                slog::info << "eyeCoordinatesRightEye("<< i << "): (x,y)=(" << eyeCoordinates[i].xright << "," << eyeCoordinates[i].yright << ")"  << slog::endl;
                // cv::imshow("Detection results", frame);
                // cv::waitKey(0);
                // cv::destroyAllWindows();
                // ---------------------------------------------------------------------------------------------------
                // // --------------------------- Do Face and Landmark Detection for eye center--------------------------
                dlib::rectangle known_det;
                dlib::matrix<dlib::rgb_pixel> enroll_chip; //original extract chip
                known_det.set_left(rect.x);
                known_det.set_top(rect.y);
                known_det.set_right(rect.x + rect.width);
                known_det.set_bottom(rect.y + rect.height);
                slog::info << "known_det("<<  known_det.left() << "," << known_det.right() << "," <<
                                 known_det.top() << "," << known_det.bottom() << ")" << slog::endl;
                std::vector<dlib::point> parts;
                parts.resize(5);
                //mapping to dlibLandmark leftEye:2 3 rightEye:1 0 nosePhiltrum:4
                parts[0].x() = rect.x + rect.width * (normed_landmarks[2 * 3]);
                parts[0].y() = rect.y + rect.height * (normed_landmarks[2 * 3 + 1]);
                parts[1].x() = rect.x + rect.width * (normed_landmarks[2 * 2]); 
                parts[1].y() = rect.y + rect.height * (normed_landmarks[2 * 2 + 1]);
                parts[2].x() = rect.x + rect.width * (normed_landmarks[2 * 1]); 
                parts[2].y() = rect.y + rect.height * (normed_landmarks[2 * 1 + 1]);
                parts[3].x() = rect.x + rect.width * (normed_landmarks[2 * 0]); 
                parts[3].y() = rect.y + rect.height * (normed_landmarks[2 * 0 + 1]);
                parts[4].x() = rect.x + rect.width * (normed_landmarks[2 * 5]); 
                parts[4].y() = rect.y + rect.height * (normed_landmarks[2 * 5 + 1]);
// std::cout<<"12"<<endl;
                for (int k = 0; k < 5; k++) {
                    cv::circle(frame, cv::Point(parts[k].x(),  parts[k].y()), 1 + static_cast<int>(0.012 * rect.width), cv::Scalar(0, 255, 255), -1);
                    string lmText = to_string(k); 
                    // cv::putText(frame, lmText, cv::Point(parts[k].x(),  parts[k].y()), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0));
                }
                cv::imshow("Detection results", frame);
                cv::waitKey(100);
                cv::destroyAllWindows();
                dlib::full_object_detection shape_local(known_det, parts);
                dlib::cv_image<dlib::rgb_pixel> cv_imgFR(frame);
                dlib::matrix<dlib::rgb_pixel> imgFR;
                assign_image(imgFR, cv_imgFR);
                dlib::extract_image_chip(imgFR, dlib::get_face_chip_details(shape_local, FR_IMAGE_HEIGHT, FR_IMAGE_PADDING*0.01), enroll_chip);
                cv::Mat chipMat = dlib::toMat(enroll_chip);
                cv::imshow("extract_image_chip", chipMat);
                cv::waitKey(100);
                cv::destroyAllWindows();
                // ---------------------------------------------------------------------------------------------------

                // --------------------------Prepare FR input---------------------------------------------------------
                if (image_size != chipMat.rows * chipMat.cols) {
                    slog::info << "FR image_size didn`t match network_input_size"<< slog::endl;
                }
                // slog::info << "dims[0]: "<< input->dims()[0] << ", dims[1]: "<< input->dims()[1] << ", dims[2]: "<< input->dims()[2]
                // << ", image_size: " << image_size << ", num_channels:" << num_channels << slog::endl;
                // std::cout<<"13"<<"chipMat.rows: "<<chipMat.rows<<", chipMat.cols: "<<chipMat.cols<<endl;
                // // unsigned char test_image[224*224*3];
                // memcpy(test_image, chipMat.data, chipMat.rows * chipMat.cols*3);
                std::memcpy(input_image, chipMat.data, chipMat.rows * chipMat.cols*3);
// std::cout<<"13"<<endl;
                /** Iterate over all input images **/
                    /** Iterate over all pixel in image (r,g,b) **/
                    for (size_t pid = 0; pid < image_size; pid++) {
                        /** Iterate over all channels **/
                        for (size_t ch = 0; ch < num_channels; ++ch) {
                            // std::cout<<"pid: "<<pid<<", ch: "<<ch<<endl;
                            data[ch *image_size + pid] = ((double)input_image[pid*num_channels + ch]- mean_values[ch]) / scale_values[ch];
                            // std::cout<<"data: "<<image_size + pid<<", input_image: "<<pid*num_channels + ch<<endl;
                        }
                    }
// std::cout<<"14"<<endl;
                // ---------------------------------------------------------------------------------------------------

                // ---------------------------FR Postprocess output blobs-----------------------------------------------
                infer_request.Infer(); //FR Do inference
// std::cout<<"14.1"<<endl;
                memset(FR_emb,0.0,512);
                memset(gender,0.0,2);
                memset(age,0.0,7);
                for (int out_c = 0; out_c < network_OutputName.size(); out_c++) {
                    const InferenceEngine::Blob::Ptr output_blob = infer_request.GetBlob(network_OutputName[out_c]);
                    float* pOt = NULL;
                    switch (OutputName_vs_index[network_OutputName[out_c]]) {
                    case 0:
                        pOt = FR_emb;
                        break;
                    case 1:
                        pOt = gender;
                        break;
                    case 2:
                        pOt = age;
                        break;
                    default:
                        slog::info << "FR output_name error"<< slog::endl;
                    }
// std::cout<<"14.2"<<endl;
                    const auto output_data = output_blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
                    /** Validating -nt value **/
                    const int resultsCnt = output_blob->size();
                    int ntop = resultsCnt;
                    // std::cout<<"resultsCnt: "<<resultsCnt<<endl;
                    for (size_t id = 0, cnt = 0; cnt < ntop; cnt++, id++) {
                        /** Getting probability for resulting class **/
                        pOt[cnt] = output_data[id];
                        // std::cout<<"ntop: "<<ntop<<", cnt: "<<cnt<<"output_data[id]"<<to_string(output_data[id])<<endl;
// std::cout<<"14.3"<<endl;
                    }
// std::cout<<"14.4"<<endl;
                }
                // -----------------------------------------------------------------------------------------------------
                // slog::info << "FR features[0,1,127,510,511]: " 
                // << "[" << FR_emb[0] << ", " << FR_emb[1] << ", " << FR_emb[127] << ", " << FR_emb[510] << ", " << FR_emb[511] << "] " << slog::endl;
            } //detected faces vector array
// std::cout<<"14.5"<<endl;
            std::vector<float> fv;
            if(prev_detection_results.size() == 0){ //for no FD found give false eyes detected bool and zero coordinates
                eyeCoordinates.push_back(EyePair(false, false, 0, 0, 0, 0));
                std::string detectFailFileName = "detectFail/" + to_string(detectFailCount) + ".jpg";
                detectFailCount ++;
                cv::imwrite(detectFailFileName, frame);
                fv.resize(FR_EMBEDDING_SIZE);
                for(int emb = 0; emb < FR_EMBEDDING_SIZE; emb++){
                    fv.push_back(0.0);
                }
                // return ReturnStatus(ReturnCode::FaceDetectionError);
            }else{
                // --------------------------- Assign 512-D embedded features vector -----------------------------------
                //std::vector<float> fv = {1.0, 2.0, 8.88, 765.88989};
                for(int emb = 0; emb < FR_EMBEDDING_SIZE; emb++){
                    fv.push_back(FR_emb[emb]);
                }
        // std::cout<<"14.6"<<endl;
                // -----------------------------------------------------------------------------------------------------
            }
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(fv.data());
            int dataSize = sizeof(float) * fv.size();
            templ.resize(dataSize);
            std::memcpy(templ.data(), bytes, dataSize);
            slog::info << "FR features size: "<<fv.size()<< " fv[0,1,127,510,511]: " 
            << "[" << fv[0] << ", " << fv[1] << ", " << fv[127] << ", " << fv[510] << ", " << fv[511] << "] " << slog::endl;
        } //faces vector size array

    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
    }
// std::cout<<"15"<<endl;
    return ReturnStatus(ReturnCode::Success);
}

ReturnStatus
NullImplFRVT11::matchTemplates(
        const std::vector<uint8_t> &verifTemplate,
        const std::vector<uint8_t> &enrollTemplate,
        double &similarity)
{
// std::cout<<"16"<<endl;
    float *featureVector = (float *)enrollTemplate.data();
    float *vfeatureVector = (float *)verifTemplate.data();

    // for (unsigned int i=0; i<this->featureVectorSize; i++) {
    //     std::cout << featureVector[i] << std::endl;
    //     std::cout << vfeatureVector[i] << std::endl;
    // }

    dlib::matrix<float, 0, 1> out_matrix;
    dlib::matrix<float, 0, 1> vout_matrix;
    out_matrix.set_size(FR_EMBEDDING_SIZE);
    vout_matrix.set_size(FR_EMBEDDING_SIZE);

    for (int j = 0; j < FR_EMBEDDING_SIZE; j++){
        out_matrix(j) = featureVector[j];
        vout_matrix(j) = vfeatureVector[j];
    }
    similarity = 1.00 - (dlib::length(out_matrix - vout_matrix)*0.50 - 0.20);

    slog::info << "out_matrix[0,1,127,510,511]: " 
    << "[" << out_matrix(0) << ", " << out_matrix(1) << ", " << out_matrix(127) << ", " << out_matrix(510) << ", " << out_matrix(511) << "] " << slog::endl;
    slog::info << "vout_matrix[0,1,127,510,511]: " 
    << "[" << vout_matrix(0) << ", " << vout_matrix(1) << ", " << vout_matrix(127) << ", " << vout_matrix(510) << ", " << vout_matrix(511) << "] " << slog::endl;
    slog::info << "similarity: " << similarity << slog::endl;
// std::cout<<"17"<<endl;
    // similarity = rand() % 1000 + 1;
    return ReturnStatus(ReturnCode::Success);
}

std::shared_ptr<Interface>
Interface::getImplementation()
{
    // std::cout<<"18"<<endl;
    return std::make_shared<NullImplFRVT11>();
}





