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

NullImplFRVT11::NullImplFRVT11() {}

NullImplFRVT11::~NullImplFRVT11() {
    // if(input_image){
    //     delete[] input_image;
    //     input_image = NULL;
    // }
    //For FD
    //release the buffer
    free(pBuffer);
    // SCOPE_EXIT{ tf_utils::DeleteGraph(graphFD); }; // Auto-delete on scope exit.
    // face_input_detector = dlib::get_frontal_face_detector();
    //For FR
    SCOPE_EXIT{ tf_utils::DeleteGraph(graph); }; // Auto-delete on scope exit.
}

ReturnStatus
NullImplFRVT11::initialize(const std::string &configDir)
{
	try {
        enrollCount = 0; 
        //FD
        //pBuffer is used in the detection functions.
        //If you call functions in multiple threads, please create one buffer for each thread!
        pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
        // std::string FDFileName = configDir + "/mtcnn_frozen_model.pb";
        // sessionFD=load_graph(FDFileName.c_str(),&graphFD);
        // if (sessionFD == nullptr) {
        //     std::cout << "[INFO] Can't load FDgraph" << std::endl;
        //     return ReturnStatus(ReturnCode::ConfigError);
        // }else{
        //     std::cout << "[INFO] Load FDgraph success" << std::endl;
        // }
        // face_input_detector = dlib::get_frontal_face_detector(); //ML
        // std::string facedetectCNNFileName = configDir + "/geo_vision_face_detect.dat";
        // dlib::deserialize(facedetectCNNFileName) >> net;  
        //LM
        std::string landMarkFileName = configDir + "/geo_vision_5_face_landmarks.dat";
        dlib::deserialize(landMarkFileName) >> sp_5; //read dlib landmark model
        m_JitterCount = FR_JITTER_COUNT;
        // if(!input_image){
        //     input_image = new unsigned char [FR_IMAGE_HEIGHT * FR_IMAGE_HEIGHT *3];
        // }

        //FR
        std::string FRFileName = configDir + "/09-02_02-45.pb";
        graph = tf_utils::LoadGraph(FRFileName.c_str());
         
        if (graph == nullptr) {
            // std::cout << "[INFO] Can't load graph" << std::endl;
            return ReturnStatus(ReturnCode::ConfigError);
        }else{
            // std::cout << "[INFO] Load graph success" << std::endl;
        }
        mean_values[0]  = mean_values[1]  = mean_values[2]  = 255.0*0.5;
	    scale_values[0] = scale_values[1] = scale_values[2] = 255.0*0.5;
        // -----------------------------------------------------------------------------------------------------
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
    try {
        clock_t begin = clock();
        for (unsigned int i=0; i<faces.size(); i++) {
            mtx.lock();
            // cout << "00" << endl;;
            // imgCount++;
            // ----------------------------------------------------------------------------------------------------
            // std::list<Face::Ptr> facesAttributes;
            size_t id = 0;
            // saveImgMtx.lock();
            cv::Mat frame = cv::Mat(faces[i].height, faces[i].width, CV_8UC3);
            cv::Mat showframe;
            cv::Mat resizeframe;
            // -------------------------------Set input data----------------------------------------------------
            // std::cout << "frvt input image height: " << faces[i].height << ", width: " << faces[i].width << ", size: " << faces[i].size() << std::endl;
            std::memcpy(frame.data, faces[i].data.get(), faces[i].size() );  
            //get resize image ratio
            float Ratio = 1.0;
            int optimalFDLength = 200;
            if(faces[i].width > optimalFDLength || faces[i].height > optimalFDLength){
                float xRatio = faces[i].width/(float)optimalFDLength;
                float yRatio = faces[i].height/(float)optimalFDLength;
                Ratio = xRatio > yRatio ? xRatio:yRatio;
            }
            // std::cout << "frvt input Ratio: " << Ratio << std::endl;
            frame.copyTo(showframe);
            dlib::matrix<dlib::rgb_pixel> enroll_chip; //original extract chip
            dlib::matrix<dlib::bgr_pixel> enroll_chipBGR; //original extract chip
            std::vector<dlib::point> parts;
            dlib::cv_image<dlib::rgb_pixel> cv_imgFR(frame);
            //use dlib to resize (avoid opencv resize uses threading)
            dlib::matrix<dlib::rgb_pixel> imgFR;
            dlib::assign_image(imgFR, cv_imgFR);
            dlib::matrix<dlib::rgb_pixel> imgFRResized(int(faces[i].height/Ratio),int(faces[i].width/Ratio));
            dlib::resize_image(imgFR,imgFRResized,dlib::interpolate_bilinear());
            cv::Mat imgResized = dlib::toMat(imgFRResized);

            int * pResults = NULL; 
            string uuidName = ProduceUUID();
            // string chipFileName = "FDresults/OriImg(" + ProduceUUID() + ").jpg";
            // dlib::save_jpeg(imgFR,chipFileName,100);
            // saveImgMtx.unlock();
            clock_t beginFD = clock();
            // std::cout<<"001: "<<std::endl;
            std::vector<dlib::rectangle> face_det;
            face_det.resize(1);
            // std::cout<<"002: "<<std::endl;


            ///////////////////////////////////////////
            // CNN face detection 
            // Best detection rate
            //////////////////////////////////////////
            //!!! The input image must be a BGR one (three-channel) instead of RGB
            //!!! DO NOT RELEASE pResults !!!

            pResults = facedetect_cnn(pBuffer, (unsigned char*)(imgResized.ptr(0)), imgResized.cols, imgResized.rows, (int)imgResized.step);
            int facesDetected = pResults ? *pResults : 0;
            cv::Mat result_image = frame.clone();
            int xleftEyeCenter = 0;
            int yleftEyeCenter = 0;
            int xRightEyeCenter = 0;
            int yRightEyeCenter = 0;
            // dlib::full_object_detection shape_5;
            // std::cout<<"003: "<<std::endl;
            //print the detection results
            for(int i = 0; i < facesDetected; i++)
            {
                short * p = ((short*)(pResults+1))+142*i;
                int confidence = p[0];
                int x = int(p[1]*Ratio);
                int y = int(p[2]*Ratio);
                int w = int(p[3]*Ratio);
                int h = int(p[4]*Ratio);
// std::cout<<"004: "<<std::endl;
                face_det[0].set_bottom(y+h);
                face_det[0].set_left(x);
                face_det[0].set_right(x+w);
                face_det[0].set_top(y);
// std::cout<<"005: "<<std::endl;
                //  std::cout << "[INFO] YSQ::face_det :[" << face_det[0].bottom() << ","<< face_det[0].left() << ","
                    // << face_det[0].right() << ","<< face_det[0].top() << "]"<< std::endl;
                
                //show the score of the face. Its range is [0-100]
                char sScore[256];
                snprintf(sScore, 256, "%d", confidence);
                cv::putText(result_image, sScore, cv::Point(x, y-3), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                //draw face rectangle
                rectangle(result_image, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
                //draw five face landmarks in different colors
                cv::circle(result_image, cv::Point(int(p[5]*Ratio), int(p[5 + 1]*Ratio)), 1, cv::Scalar(255, 0, 0), 2);
                cv::circle(result_image, cv::Point(int(p[5 + 2]*Ratio), int(p[5 + 3]*Ratio)), 1, cv::Scalar(0, 0, 255), 2);
                cv::circle(result_image, cv::Point(int(p[5 + 4]*Ratio), int(p[5 + 5]*Ratio)), 1, cv::Scalar(0, 255, 0), 2);
                cv::circle(result_image, cv::Point(int(p[5 + 6]*Ratio), int(p[5 + 7]*Ratio)), 1, cv::Scalar(255, 0, 255), 2);
                cv::circle(result_image, cv::Point(int(p[5 + 8]*Ratio), int(p[5 + 9]*Ratio)), 1, cv::Scalar(0, 255, 255), 2);
                
                cv::putText(result_image, "0", cv::Point(int(p[5]*Ratio), int(p[5 + 1]*Ratio)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                cv::putText(result_image, "1", cv::Point(int(p[5 + 2]*Ratio), int(p[5 + 3]*Ratio)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                cv::putText(result_image, "2", cv::Point(int(p[5 + 4]*Ratio), int(p[5 + 5]*Ratio)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                cv::putText(result_image, "3", cv::Point(int(p[5 + 6]*Ratio), int(p[5 + 7]*Ratio)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                cv::putText(result_image, "4", cv::Point(int(p[5 + 8]*Ratio), int(p[5 + 9]*Ratio)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);


                // int eyeWidth = int(abs((p[5]*2 - p[5 + 2]))*0.25)
                // shape_5.part(0).x() = ;
                // shape_5.part(0).y() = ;
                // shape_5.part(1).x() = ;
                // shape_5.part(1).y() = ;
                // shape_5.part(2).x() = p[5]*2 - eyeWidth;
                // shape_5.part(2).y() = ;
                // shape_5.part(3).x() = ;
                // shape_5.part(3).y() = ;
                // shape_5.part(4).x() = ;
                // shape_5.part(4).y() = ;


// std::cout<<"006: "<<std::endl;
                xleftEyeCenter = int(p[5]*Ratio);
                yleftEyeCenter = int(p[5 + 1]*Ratio);
                xRightEyeCenter = int(p[5 + 2]*Ratio);
                yRightEyeCenter = int(p[5 + 3]*Ratio);
                //print the result
                // printf("face %d: confidence=%d, [%d, %d, %d, %d] (%d,%d) (%d,%d) (%d,%d) (%d,%d) (%d,%d)\n", 
                //         i, confidence, x, y, w, h, 
                //         p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13],p[14]);

                // saveImgMtx.lock();
                // string detectFileName = "FDresults/face(" + uuidName + ").jpg";
                // dlib::cv_image<dlib::rgb_pixel> cv_temp(result_image);
                // dlib::matrix<dlib::rgb_pixel> dlib_array2d;
                // dlib::assign_image(dlib_array2d, cv_temp);
                // dlib::save_jpeg(dlib_array2d,detectFileName,100);
                // saveImgMtx.unlock();
// std::cout<<"007: "<<std::endl;
            }
            // if(facesDetected==0){
            //     saveImgMtx.lock();
            //     string detectFileName = "FDresults/fail(" + uuidName + ").jpg";
            //     dlib::cv_image<dlib::rgb_pixel> cv_temp(result_image);
            //     dlib::matrix<dlib::rgb_pixel> dlib_array2d;
            //     dlib::assign_image(dlib_array2d, cv_temp);
            //     dlib::save_jpeg(dlib_array2d,detectFileName,100);
            //     saveImgMtx.unlock();
            // }

            // std::cout<<"008: "<<std::endl;

            clock_t endFD = clock();
            double time_spentFD = (double)(endFD - beginFD) / CLOCKS_PER_SEC;
            // std::cout << "[INFO] FD YSQ execute time: "<<time_spentFD<< " sec spent" << std::endl;
            // std::vector<dlib::rectangle> face_det = face_input_detector(imgFR);
            // For multi detected face
            int maxFaceId = 0;
            int maxRectArea = 0;
            faceDetectCount = facesDetected;
            // if(faceDetectCount == 0){
            //     // dlib::matrix<dlib::rgb_pixel> imgFRPry;
            //     // dlib::assign_image(imgFRPry,imgFR);
            //     // dlib::pyramid_up(imgFRPry);
            //     std::vector<dlib::rectangle> face_det = face_input_detector(imgFR);
            //     faceDetectCount = face_det.size();
            //     std::cout << "[INFO] dlib::cnnFD used faceDetectCount:" << faceDetectCount<< std::endl;
            //     if(faceDetectCount==1){
            //         clock_t beginCnnFD = clock();
            //         const std::vector<dlib::mmod_rect> cnnFD_det = net(imgFR);
            //         clock_t endCnnFD = clock();
            //         double time_spentFD = (double)(endCnnFD - beginCnnFD) / CLOCKS_PER_SEC;
            //         std::cout << "[INFO] FD dlib DL execute time: "<<time_spentFD<< " sec spent" << std::endl;
            //         std::cout << "[INFO] dlib::face_det pre:[" << cnnFD_det[0].rect.top() << ","<< cnnFD_det[0].rect.left() << ","
            //         << cnnFD_det[0].rect.right() << ","<< cnnFD_det[0].rect.bottom() << "]"<< std::endl;
            //         face_det[0].set_bottom(cnnFD_det[0].rect.top());
            //         face_det[0].set_left(cnnFD_det[0].rect.left());
            //         face_det[0].set_right(cnnFD_det[0].rect.right());
            //         face_det[0].set_top(cnnFD_det[0].rect.bottom());
            //         // face_det[0].set_bottom(int(cnnFD_det[0].rect.top()*xRatio));
            //         // face_det[0].set_left(int(cnnFD_det[0].rect.left()*yRatio));
            //         // face_det[0].set_right(int(cnnFD_det[0].rect.right()*yRatio));
            //         // face_det[0].set_top(int(cnnFD_det[0].rect.bottom()*xRatio));
            //         std::cout << "[INFO] dlib::imgFR size:[" << imgFR.nc()<<","<< imgFR.nr()<<"]"<< std::endl;
            //         // face_det[0].set_bottom(int(face_det[0].bottom()*0.5));
            //         // face_det[0].set_left(int(face_det[0].left()*0.5));
            //         // face_det[0].set_right(int(face_det[0].right()*0.5));
            //         // face_det[0].set_top(int(face_det[0].top()*0.5));
            //         // std::cout << "[INFO] dlib::face_det post:[" << face_det[0].bottom() << ","<< face_det[0].left() << ","
            //         // << face_det[0].right() << ","<< face_det[0].top() << "]"<< std::endl;
            //     }
            // }
            if(faceDetectCount > 0){
                // cout << "031" << endl;
                //     if(faceDetectCount > 1){
                //     for (size_t j = 0; j < faceDetectCount; j++) {
                //         if(face_det[j].width() * face_det[j].height() >  maxRectArea){
                //             maxRectArea = face_det[j].width() * face_det[j].height();
                //             maxFaceId = j;
                //         }
                //     }
                // }
                //====================Do dlib Landmark====================================
                dlib::full_object_detection shape_5 = sp_5(imgFR, face_det[0]);
                cv::Point pt1(face_det[maxFaceId].left(), face_det[maxFaceId].top());
                // and its bottom right corner.
                cv::Point pt2(face_det[maxFaceId].right(), face_det[maxFaceId].bottom());
                // These two calls...
                cv::rectangle(showframe, pt1, pt2, cv::Scalar(0, 0, 255));
// std::cout<<"009: "<<std::endl;
                // --------------------------- Assign Landmark for eye center----------------------------
                //dlibLandmark leftEye:2 3 rightEye:1 0 nosePhiltrum:4
                // cv::putText(result_image, "0", cv::Point(p[5]*2, p[5 + 1]*2), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                // cv::putText(result_image, "1", cv::Point(p[5 + 2]*2, p[5 + 3]*2), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                // int xleftEyeCenter = int ((shape_5.part(2).x() + shape_5.part(3).x())*0.5);
                // int yleftEyeCenter = int ((shape_5.part(2).y() + shape_5.part(3).y())*0.5);
                cv::circle(showframe, cv::Point(xleftEyeCenter, yleftEyeCenter), 1 + static_cast<int>(0.012 * face_det[maxFaceId].width()), cv::Scalar(255, 0, 0), -1);
                // int xRightEyeCenter = int ((shape_5.part(0).x() + shape_5.part(1).x())*0.5);
                // int yRightEyeCenter = int ((shape_5.part(0).y() + shape_5.part(1).y())*0.5);
                cv::circle(showframe, cv::Point(xRightEyeCenter, yRightEyeCenter), 1 + static_cast<int>(0.012 * face_det[maxFaceId].width()), cv::Scalar(0, 0, 255), -1);
                eyeCoordinates.clear();
                eyeCoordinates.shrink_to_fit();
                eyeCoordinates.push_back(EyePair(true, true, xRightEyeCenter, yRightEyeCenter, xleftEyeCenter, yleftEyeCenter)); //left right eyes are mirrored
                //////////ISO standard: The label "left" refers to subject's left eye (and similarly for the right eye), such that xright < xleft/////////////////
                // cout << "eyeCoordinatesLeftEye("<< i << "): (x,y)=(" << eyeCoordinates[i].xleft << "," << eyeCoordinates[i].yleft << ")"  << endl;
                // cout << "eyeCoordinatesRightEye("<< i << "): (x,y)=(" << eyeCoordinates[i].xright << "," << eyeCoordinates[i].yright << ")"  << endl;
                // saveImgMtx.lock();
                // string detectFileName = "FDresults/face(" + ProduceUUID() + ").jpg";
                // dlib::cv_image<dlib::rgb_pixel> cv_temp(showframe);
                // dlib::matrix<dlib::rgb_pixel> dlib_array2d;
                // dlib::assign_image(dlib_array2d, cv_temp);
                // dlib::save_jpeg(dlib_array2d,detectFileName,100);
                // cv::imwrite(detectFileName, showframe);
                // saveImgMtx.unlock();
                // std::cout<<"010: "<<std::endl;
                //-------------------------------Crop face image================================
                dlib::extract_image_chip(imgFR, dlib::get_face_chip_details(shape_5, FR_IMAGE_HEIGHT, FR_IMAGE_PADDING*0.01), enroll_chip);
                dlib::assign_image(enroll_chipBGR, enroll_chip);
                // if(enrollCount==0){
                    // cv::Mat enrollChipMat = dlib::toMat(enroll_chip);
                //     saveImgMtx.lock();
                //     std::time_t t = std::time(0);   // get time now
                //     std::tm* now = std::localtime(&t);
                //     std::srand((unsigned) time(&t));
                //     int rndNumber = rand() % 10000;
                //     string chipFileName = "FDresults/chip(" + uuidName + ").jpg";
                //     dlib::save_jpeg(enroll_chip,chipFileName,100);
                // // }
                // // cv::imwrite(chipFileName, enrollChipMat);
                // saveImgMtx.unlock();
                // cv::waitKey(300);
                // cv::destroyAllWindows();
                // cout << "06" << endl;
                
            }else{//no face detected return error enum
                eyeCoordinates.clear();
                eyeCoordinates.shrink_to_fit();
                eyeCoordinates.push_back(EyePair(false, false, xRightEyeCenter, yRightEyeCenter, xleftEyeCenter, yleftEyeCenter));
                // return ReturnStatus(ReturnCode::FaceDetectionError);
            }            
            

            std::vector<dlib::matrix<float, 0, 1>> SVM_descriptor;
            std::vector<dlib::matrix<dlib::rgb_pixel>> SVM_distrub_color_crops;
            int cropsCount = m_JitterCount;
            if(faceDetectCount > 0){
                if(role == TemplateRole::Enrollment_11 || role == TemplateRole::Enrollment_1N){
                    // slog::info << "FR image TemplateRole Enrollment"<< slog::endl;
                    if(m_JitterCount > 0){
                        SVM_distrub_color_crops = this->jitter_image(enroll_chip, FR_IMAGE_HEIGHT, FR_IMAGE_HEIGHT);
                    }else{
                        SVM_distrub_color_crops.push_back(enroll_chip);
                        cropsCount = 1;
                    }
                }else{
                    // slog::info << "FR image TemplateRole Verification"<< slog::endl;
                    SVM_distrub_color_crops.push_back(enroll_chip);
                    cropsCount = 1;
                }
            }else{
                cropsCount = 0;
            }


            // std::cout<<"011: "<<std::endl;

            for (int i = 0; i < cropsCount; i++)
            {
                // cv::Mat image = dlib::toMat(SVM_distrub_color_crops[i]);

                // std::vector<TF_Output> 	input_tensors, output_tensors;
                // std::vector<TF_Tensor*> input_values, output_values;

                // // input tensor shape.
                // int num_dims = 4;
                // std::int64_t input_dims[4] = {1, image.rows, image.cols, 3}; //1 is number of batch, and 3 is the no of channels.
                // int num_bytes_in = image.cols * image.rows * 3; //3 is the number of channels.
                
                // input_tensors.push_back({TF_GraphOperationByName(graph, "input"),0});
                // input_values.push_back(TF_NewTensor(TF_UINT8, input_dims, num_dims, image.data, num_bytes_in, &Deallocator, 0));

                // output_tensors.push_back({ TF_GraphOperationByName(graph, "embedding"),0 });
                // output_values.push_back(nullptr);

                // TF_Status* status = TF_NewStatus();
                // TF_SessionOptions* options = TF_NewSessionOptions();
                // std::array<std::uint8_t, 13> config = {{ 0x0a ,0x07, 0x0a, 0x03, 0x43, 0x50, 0x55, 0x10, 0x01, 0x10, 0x01, 0x28, 0x01}};
                // TF_SetConfig(options, config.data(), config.size(), status);
                // TF_Session* session = TF_NewSession(graph, options, status);
                // TF_DeleteSessionOptions(options);
                // TF_SessionRun(session, nullptr,
                //     &input_tensors[0], &input_values[0], input_values.size(),
                //     &output_tensors[0], &output_values[0], 1, //1 is the number of outputs count..
                //     nullptr, 0, nullptr, status
                // );
                // if (TF_GetCode(status) != TF_OK)
                // {
                //     printf("ERROR: SessionRun: %s", TF_Message(status));
                // }
                // TF_DeleteStatus(status);

                // float* embeddingResults = new float[FR_EMBEDDING_SIZE];
                // embeddingResults = static_cast<float_t*>(TF_TensorData(output_values[0]));
                // auto data = static_cast<float*>(TF_TensorData(output_values[0]));

                // std::cout << "Output vals: " << data[0] << ", " << data[1] << ", " << data[2] << ", " << data[3] << std::endl;

                // cout<<typeid(embeddingResults).name() <<endl;

                // if(i == cropsCount - 1){
                //     std::string jitterShowName = "LastChip(" + to_string(i) + ")";
                //     cv::imshow(jitterShowName, chipMat);
                //     saveImgMtx.lock();
                //     // std::time_t t = std::time(0);   // get time now
                //     // std::tm* now = std::localtime(&t);
                //     // srand((unsigned) time(&t));
                //     // int rndNumber = rand() % 10000;
                //     // string chipFileName = "FDresults/chip(" + to_string(now->tm_year + 1900) + "_"
                //     // + to_string(now->tm_mon + 1) + "_"  + to_string(now->tm_mday) + "_" + to_string(now->tm_hour) + "_" 
                //     // + to_string(now->tm_min) + "_" + to_string(now->tm_sec) + "_" + to_string(rndNumber) + ").jpg"; 
                //     string chipFileName = "FDresults/chip(" + ProduceUUID() + ").jpg";
                //     cv::imwrite(chipFileName, chipMat);
                //     saveImgMtx.unlock();
                //     cv::waitKey(300);
                //     cv::destroyAllWindows();
                // }
                // ---------------------------------------------------------------------------------------------------

                // --------------------------Prepare FR input---------------------------------------------------------

                // cv::Mat chipMat = dlib::toMat(SVM_distrub_color_crops[i]);
                cv::Mat chipMat = dlib::toMat(enroll_chipBGR);
                // cv::imshow("chipMat",chipMat);
                // std::memcpy(input_image, chipMat.data, chipMat.rows * chipMat.cols*3);
                // cv::waitKey();
                std::vector<float> input_data;

                cv::Mat image32f;
                chipMat.convertTo( image32f, CV_32F );
                input_data.assign( (float*) image32f.data, (float*) image32f.data + image32f.total() * image32f.channels() );
                // std::cout<<"input_data: "<<input_data.size()<<std::endl;
                float normalizeInput = 255.0*0.5;
                for(int i = 0; i< 224*224*3; i++){
                    // std::cout<<" input_dataPre: "<<input_data[i];
                    input_data[i]=float(input_data[i] - normalizeInput)/normalizeInput;
                    // std::cout<<" input_dataPost: "<<input_data[i]<<std::endl;
                }
// std::cout<<"012: "<<std::endl;
                // dimensions
                const std::vector<std::int64_t> input_dims = { 1, 224,224,3 };
                // Tensors:
                const std::vector<TF_Output> input_ops = { {TF_GraphOperationByName( graph, "input" ), 0} };
                const std::vector<TF_Tensor*> input_tensors = { tf_utils::CreateTensor( TF_FLOAT, input_dims, input_data ) };
                SCOPE_EXIT{ tf_utils::DeleteTensors(input_tensors); }; // Auto-delete on scope exit.

                const std::vector<TF_Output> out_ops = { {TF_GraphOperationByName( graph, "embedding" ), 0} };
                std::vector<TF_Tensor*> output_tensors = { nullptr };
                SCOPE_EXIT{ tf_utils::DeleteTensors(output_tensors); }; // Auto-delete on scope exit.

                // create TF session:
                TF_Status* status = TF_NewStatus();
                TF_SessionOptions* options = TF_NewSessionOptions();
                std::array<std::uint8_t, 13> config = {{ 0x0a ,0x07, 0x0a, 0x03, 0x43, 0x50, 0x55, 0x10, 0x01, 0x10, 0x01, 0x28, 0x01}};
                TF_SetConfig(options, config.data(), config.size(), status);
                TF_Session* session = tf_utils::CreateSession( graph, options, status );
                // run Session
                clock_t beginFR = clock();
                const TF_Code code = tf_utils::RunSession( session, input_ops, input_tensors, out_ops, output_tensors );
                clock_t endFR = clock();
                double time_spentFD = (double)(endFR - beginFR) / CLOCKS_PER_SEC;
                // std::cout << "[INFO] FR TF execute time: "<<time_spentFD<< " sec spent" << std::endl;
                SCOPE_EXIT{ tf_utils::DeleteSession(session); }; // Auto-delete on scope exit.
                // get the data:
                const std::vector<std::vector<float>> dataOutputResults = tf_utils::GetTensorsData<float>( output_tensors );
                // cout<< "dataOutputResults.size : "<< dataOutputResults.size()<<", dataOutputResults[0].size" << dataOutputResults[0].size()<<endl;
// std::cout<<"013: "<<std::endl;

                // std::vector<TF_Output> 	input_tensors, output_tensors;
                // std::vector<TF_Tensor*> input_values, output_values;

                // //input tensor shape.
                // int num_dims = 4;
                // std::int64_t input_dims[4] = {1, chipMat.rows, chipMat.cols, 3}; //1 is number of batch, and 3 is the no of channels.
                // int num_bytes_in = chipMat.cols * chipMat.rows * 3; //3 is the number of channels.
                
                // input_tensors.push_back({TF_GraphOperationByName(graph, "input"),0});
                // input_values.push_back(TF_NewTensor(TF_FLOAT, input_dims, num_dims, chipMat.data, num_bytes_in, &Deallocator, 0));

                // output_tensors.push_back({ TF_GraphOperationByName(graph, "embedding"),0 });
                // output_values.push_back(nullptr);


                // // create TF session:
                // TF_Status* status = TF_NewStatus();
                // TF_SessionOptions* options = TF_NewSessionOptions();
                // std::array<std::uint8_t, 13> config = {{ 0x0a ,0x07, 0x0a, 0x03, 0x43, 0x50, 0x55, 0x10, 0x01, 0x10, 0x01, 0x28, 0x01}};
                // TF_SetConfig(options, config.data(), config.size(), status);
                // TF_Session* session = tf_utils::CreateSession( graph, options, status );

                // TF_SessionRun(session, nullptr,
                //     &input_tensors[0], &input_values[0], input_values.size(),
                //     &output_tensors[0], &output_values[0], 1, //1 is the number of outputs count..
                //     nullptr, 0, nullptr, status
                // );
                // if (TF_GetCode(status) != TF_OK)
                // {
                //     printf("ERROR: SessionRun: %s", TF_Message(status));
                // }
                
                // auto detection_classes = static_cast<float_t*>(TF_TensorData(output_values[0]));




                // ---------------------------------------------------------------------------------------------------
                ////deprecated intel inference engine code////
                // --------------------------Prepare FR input---------------------------------------------------------
                // if (image_size != chipMat.rows * chipMat.cols) {
                //     slog::info << "FR image_size didn`t match network_input_size"<< slog::endl;
                // }
                // slog::info << "dims[0]: "<< input->dims()[0] << ", dims[1]: "<< input->dims()[1] << ", dims[2]: "<< input->dims()[2]
                // << ", image_size: " << image_size << ", num_channels:" << num_channels << slog::endl;
                // std::cout<<"13"<<"chipMat.rows: "<<chipMat.rows<<", chipMat.cols: "<<chipMat.cols<<endl;
                // // unsigned char test_image[224*224*3];
                // memcpy(test_image, chipMat.data, chipMat.rows * chipMat.cols*3);
                // std::memcpy(input_image, chipMat.data, chipMat.rows * chipMat.cols*3);
                // /** Iterate over all input images **/
                //     /** Iterate over all pixel in image (r,g,b) **/
                //     for (size_t pid = 0; pid < image_size; pid++) {
                //         /** Iterate over all channels **/
                //         for (size_t ch = 0; ch < num_channels; ++ch) {
                //             // std::cout<<"pid: "<<pid<<", ch: "<<ch<<endl;
                //             data[ch *image_size + pid] = ((double)input_image[pid*num_channels + ch]- mean_values[ch]) / scale_values[ch];
                //             // std::cout<<"data: "<<image_size + pid<<", input_image: "<<pid*num_channels + ch<<endl;
                //         }
                //     }
                // ---------------------------------------------------------------------------------------------------

                // ---------------------------FR Postprocess output blobs-----------------------------------------------
            //         infer_request.Infer(); //FR Do inference
                    memset(jitterFR_emb,0.0,FR_EMBEDDING_SIZE*__SIZEOF_FLOAT__);
                    for(int i=0;i<FR_EMBEDDING_SIZE;i++){
                        jitterFR_emb[i] = dataOutputResults[0][i];
                    }
// std::cout<<"014: "<<std::endl;
                    // float sum = 0;
                    // float map[FR_EMBEDDING_SIZE];
                    // memset(map,0.0,FR_EMBEDDING_SIZE*__SIZEOF_FLOAT__);
                    // for(int i=0;i<FR_EMBEDDING_SIZE;i++){
                    //     map[i] = data[0][i];
                    // }
                    // for (int j = 0; j < FR_EMBEDDING_SIZE; j++) { sum = sum + map[j] * map[j]; }
                    // sum = sqrt(sum);
                    // for (int j = 0; j < FR_EMBEDDING_SIZE; j++) { jitterFR_emb[j] = map[j] / sum; }



                    // for(int i=0;i<FR_EMBEDDING_SIZE;i++){
                    //     jitterFR_emb[i] = embeddingResults[0][i];
                    // }
            //         memset(gender,0.0,2);
            //         memset(age,0.0,7);
                    // for (int out_c = 0; out_c < network_OutputName.size(); out_c++) {
            //             const InferenceEngine::Blob::Ptr output_blob = infer_request.GetBlob(network_OutputName[out_c]);
            //             float* pOt = NULL;
            //             switch (OutputName_vs_index[network_OutputName[out_c]]) {
            //             case 0:
            //                 pOt = jitterFR_emb;
            //                 break;
            //             case 1:
            //                 pOt = gender;
            //                 break;
            //             case 2:
            //                 pOt = age;
            //                 break;
            //             default:
            //                 std::string FRmsg = "FR output_name error";
            //                 // slog::info << FRmsg << slog::endl;
            //             }
            //             const auto output_data = output_blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
            //             /** Validating -nt value **/
            //             const int resultsCnt = output_blob->size();
            //             int ntop = resultsCnt;
            //             // std::cout<<"resultsCnt: "<<resultsCnt<<endl;
            //             for (size_t id = 0, cnt = 0; cnt < ntop; cnt++, id++) {
            //                 /** Getting probability for resulting class **/
            //                 pOt[cnt] = output_data[id];
            //                 // std::cout<<"ntop: "<<ntop<<", cnt: "<<cnt<<"output_data[id]"<<to_string(output_data[id])<<endl;
            //             }
            //         }
                    SVM_descriptor.push_back(array_to_dlib_1D_matrix(1, jitterFR_emb, FR_EMBEDDING_SIZE)[0]);
            //         // FR_emb[emb]
            } //jitter cropsCount
            //     // -----------------------------------------------------------------------------------------------------
                dlib::matrix<float, 0, 1> temp_mat = mean(mat(SVM_descriptor));
                // std::vector<dlib::matrix<float, 0, 1>> EnrollDescriptor;
                // cout << "dlib::length(temp_mat): " << dlib::length(temp_mat) << std::endl;
                // int normalizeLength = dlib::length(temp_mat) < 1 ? 1 : dlib::length(temp_mat);
                // cout << "normalizeLength: " << normalizeLength << std::endl;
                // int normalizeLength = dlib::length(temp_mat);
                // EnrollDescriptor.push_back(temp_mat / normalizeLength); //Use jitter image and normalize to length
                memset(FR_emb,0.0,FR_EMBEDDING_SIZE);
                for (int j = 0; j < FR_EMBEDDING_SIZE; j++)
                {
                    FR_emb[j] = temp_mat(j, 0);
                    // FR_emb[j] = EnrollDescriptor[EnrollDescriptor.size() - 1](j, 0);
                }
                // std::vector <dlib::matrix<float, 0, 1>>().swap(EnrollDescriptor);

                // std::cout << "FR features[0,1,127,510,511]: " 
                // << "[" << FR_emb[0] << ", " << FR_emb[1] << ", " << FR_emb[127] << ", " << FR_emb[510] << ", " << FR_emb[511] << "] " << std::endl;

            // } //detected faces vector array
// std::cout<<"015: "<<std::endl;
            std::vector<float> fv;
            if(faceDetectCount == 0){ //for no FD found give false eyes detected bool and zero coordinates
                eyeCoordinates.push_back(EyePair(false, false, 0, 0, 0, 0));
                // saveImgMtx.lock();
                // std::time_t t = std::time(0);   // get time now
                // std::tm* now = std::localtime(&t);
                // srand((unsigned) time(&t));
                // int rndNumber = rand() % 10000;
                // string detectFailFileName = "detectFail/face(" + to_string(now->tm_year + 1900) + "_"
                // + to_string(now->tm_mon + 1) + "_"  + to_string(now->tm_mday) + "_" + to_string(now->tm_hour) + "_" 
                // + to_string(now->tm_min) + "_" + to_string(now->tm_sec) + "_" + to_string(rndNumber) + ").jpg"; 
                // string detectFailFileName = "detectFail/face(" + ProduceUUID() + ").jpg"; 
                // cv::imwrite(detectFailFileName, frame);
                // saveImgMtx.unlock();
                // detectFailCount ++;
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
                // -----------------------------------------------------------------------------------------------------
            }
            //==test use==
            // if(enrollCount==0){
            //     std::cout<<"fv features: {  "<< std::endl;
            //     for(int i=0;i<FR_EMBEDDING_SIZE;i++){ std::cout<<fv[i] << std::endl; }
            //     std::cout<<"  }" << std::endl;
            // }
            // enrollCount=1;

            // memset(FR_emb,0,__SIZEOF_FLOAT__*FR_EMBEDDING_SIZE);
            // for(int emb = 0; emb < FR_EMBEDDING_SIZE; emb++){
            //     fv.push_back(FR_emb[emb]);
            // }
            //==test use==
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(fv.data());
            int dataSize = sizeof(float) * fv.size();
            templ.resize(dataSize);
            std::memcpy(templ.data(), bytes, dataSize);
            // std::cout << "FR features size: "<<fv.size()<< " fv[0,1,127,510,511]: " 
            // << "[" << fv[0] << ", " << fv[1] << ", " << fv[127] << ", " << fv[510] << ", " << fv[511] << "] " << std::endl;
            mtx.unlock();
            // test_matrix1.set_size(FR_EMBEDDING_SIZE);
            // test_matrix2.set_size(FR_EMBEDDING_SIZE);
            // for (int j = 0; j < FR_EMBEDDING_SIZE; j++){
            //     if(enrollCount%2==0){
            //         test_matrix1(j) = fv[j];
            //     }else{
            //         test_matrix2(j) = fv[j];
            //     }
            // }
            // if(enrollCount>0){
            //     std::cout << "FR features "<< " test_matrix1[0,1,127,510,511]: " 
            //     << "[" << test_matrix1(0) << ", " << test_matrix1(1) << ", " << test_matrix1(127) << ", " << test_matrix1(510) << ", " << test_matrix1(511) << "] " << std::endl;
            //     std::cout << "FR features "<< " test_matrix2[0,1,127,510,511]: " 
            //     << "[" << test_matrix2(0) << ", " << test_matrix2(1) << ", " << test_matrix2(127) << ", " << test_matrix2(510) << ", " << test_matrix2(511) << "] " << std::endl;
            //     // double similarity = 1.00 - (dlib::length(test_matrix1 - test_matrix2));
            //     double similarity = 1.00 - (dlib::length(test_matrix1 - test_matrix2)*0.50 - 0.20);
            //     std::cout << "FR similarity: ("<< similarity << ")enrollCount: (" << enrollCount << ") " << std::endl;
            // }
            // enrollCount++;

            //deallocate vectors
            // std::vector <dlib::matrix<float, 0, 1>>().swap(EnrollDescriptor);
            std::vector <dlib::matrix<dlib::rgb_pixel>>().swap(SVM_distrub_color_crops);
            std::vector <dlib::matrix<float, 0, 1>>().swap(SVM_descriptor);
            std::vector <dlib::point>().swap(parts);
            std::vector <dlib::rectangle>().swap(face_det);
// std::cout<<"016: "<<std::endl;
        } //faces vect`or size array
        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        // std::cout << "[INFO] NIST createTemplate execute time: "<<time_spent<< " sec spent" << std::endl;
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
        // return ReturnStatus(ReturnCode::UnknownError);
    }
    // if(faceDetectCount < 1){
        // return ReturnStatus(ReturnCode::FaceDetectionError);
    // }else{
        // std::cout << "faceDetectCount: " << faceDetectCount <<std::endl;

        // if(faceDetectCount < 1){
            // std::cout << "faceDetectCount: " << faceDetectCount << std::endl;
            //     std::cout << "templ: " << std::endl;
            // for (int j = 0; j < FR_EMBEDDING_SIZE; j++){
            //     std::cout << templ[j] << ", ";
            //     if(j%10==0){
            //         std::cout << std::endl;
            //     }
            // }
        // }

        return ReturnStatus(ReturnCode::Success);
    // }
}

ReturnStatus
NullImplFRVT11::matchTemplates(
        const std::vector<uint8_t> &verifTemplate,
        const std::vector<uint8_t> &enrollTemplate,
        double &similarity)
{
    float *featureVector = (float *)enrollTemplate.data();
    float *vfeatureVector = (float *)verifTemplate.data();
    // dlib::matrix<float, 0, 1> out_matrix;
    // dlib::matrix<float, 0, 1> vout_matrix;
    // dlib::matrix<float, 0, 1> zero_matrix;
    // out_matrix.set_size(FR_EMBEDDING_SIZE);
    // vout_matrix.set_size(FR_EMBEDDING_SIZE);
    // zero_matrix.set_size(FR_EMBEDDING_SIZE);

    float confidence = 0;
    for (int j = 0; j < FR_EMBEDDING_SIZE; j++){
        confidence = confidence + std::abs(featureVector[j] - vfeatureVector[j]);
    }
    similarity = confidence;



    // for (int j = 0; j < FR_EMBEDDING_SIZE; j++){
    //     out_matrix(j) = featureVector[j];
    //     vout_matrix(j) = vfeatureVector[j];
    //     zero_matrix(j) = 0.0;
    // }
    // float confidence = 1.00 - (dlib::length(out_matrix - vout_matrix)*0.50 - 0.20);
     

    // std::cout << "out_matrix: " << std::endl;
    // for (int j = 0; j < FR_EMBEDDING_SIZE; j++){
    //     std::cout << out_matrix(j) << ", ";
    //     if(j%10==0){
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << std::endl << "vout_matrix: " << std::endl;
    // for (int j = 0; j < FR_EMBEDDING_SIZE; j++){
    //     std::cout << vout_matrix(j) << ", ";
    //     if(j%10==0){
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << std::endl;
    // std::cout << "out_matrix length: " << dlib::length(out_matrix - zero_matrix)<<std::endl;
    // std::cout << "vout_matrix length: " << dlib::length(vout_matrix - zero_matrix)<<std::endl;

    // bool featureVectorIsAllZero,vfeatureVectorIsAllZero;
    // if( dlib::length(out_matrix - zero_matrix) == 0.0) { featureVectorIsAllZero = true; }else{ featureVectorIsAllZero = false; }
    // if( dlib::length(vout_matrix - zero_matrix) == 0.0) { vfeatureVectorIsAllZero = true; }else{ vfeatureVectorIsAllZero = false; }
    // if(featureVectorIsAllZero || vfeatureVectorIsAllZero || confidence < 0.0){ confidence = 0.0; }
    // if(confidence > 1.0 ){ confidence = 1.0; }

    // similarity = 1.00 - (dlib::length(out_matrix - vout_matrix));
    // std::cout << "out_matrix[0,1,127,510,511]: " 
    // << "[" << out_matrix(0) << ", " << out_matrix(1) << ", " << out_matrix(127) << ", " << out_matrix(510) << ", " << out_matrix(511) << "] " << std::endl;
    // std::cout << "vout_matrix[0,1,127,510,511]: " 
    // << "[" << vout_matrix(0) << ", " << vout_matrix(1) << ", " << vout_matrix(127) << ", " << vout_matrix(510) << ", " << vout_matrix(511) << "] " << std::endl;
    // std::cout << "similarity: " << similarity << std::endl;
    // similarity = rand() % 1000 + 1;
    // similarity = confidence;
    return ReturnStatus(ReturnCode::Success);
}

std::shared_ptr<Interface>
Interface::getImplementation()
{
    return std::make_shared<NullImplFRVT11>();
}

std::vector<dlib::matrix<dlib::rgb_pixel>> NullImplFRVT11::jitter_image(const dlib::matrix<dlib::rgb_pixel>& img, int height, int width)
{
    bool Jitter_num_svm[3];
    int m_disturb_gamma_svm[3];
    int m_disturb_color_svm[3];
    Jitter_num_svm[0] = Jitter_num_svm[1] = Jitter_num_svm[2] = 0;
    m_disturb_gamma_svm[0] = m_disturb_gamma_svm[1] = m_disturb_gamma_svm[2] = 5;
    m_disturb_color_svm[0] = m_disturb_color_svm[1] = m_disturb_color_svm[2] = 2;
    Jitter_num_svm[0] = 1; 
    // slog::info << "Enroll template use jitter m_JitterCount: " << m_JitterCount << slog::endl;

	crops.clear();
	crops.shrink_to_fit();
	std::vector <dlib::matrix<dlib::rgb_pixel>>().swap(crops);

	int Jitter_num_svm_count = 0;
	for (int j = 0; j < 3; j++) 
	{
		if (Jitter_num_svm[j]) 
		{
			// All this function does is make 100 copies of img, all slightly jittered by being
			// zoomed, rotated, and translated a little bit differently.
			thread_local dlib::random_cropper cropper;
			cropper.set_chip_dims(height, width);
			cropper.set_randomly_flip(true);
			cropper.set_max_object_size(0.99999);
			cropper.set_background_crops_fraction(0);
			cropper.set_min_object_size(FR_IMAGE_HEIGHT,FR_IMAGE_HEIGHT);
			cropper.set_translate_amount(0.02);
			cropper.set_max_rotation_degrees(3);
			std::vector<dlib::mmod_rect> raw_boxes(1), ignored_crop_boxes;
			raw_boxes[0] = shrink_rect(get_rect(img), 3);
			dlib::matrix<dlib::rgb_pixel> temp;
			char strpath[512];
			char FRdir_msg[512];
			for (int i = 0; i < m_JitterCount; ++i)
			{
				cropper(img, raw_boxes, temp, ignored_crop_boxes);
				crops.push_back(move(temp));
			}
			time_t seed;
			dlib::rand rnd(time(0) + seed);
			for (auto&& crop : crops)
			{
				disturb_colors(crop, rnd, m_disturb_gamma_svm[j]*0.1, m_disturb_color_svm[j]*0.1);
			}
			raw_boxes.clear();
			raw_boxes.shrink_to_fit();
			std::vector <dlib::mmod_rect>().swap(raw_boxes);
			ignored_crop_boxes.clear();
			ignored_crop_boxes.shrink_to_fit();
			std::vector <dlib::mmod_rect>().swap(ignored_crop_boxes);
			Jitter_num_svm_count++;
		}
	}
	m_JitterCount = m_JitterCount*Jitter_num_svm_count;
	if (m_JitterCount==0 || (!Jitter_num_svm[0] && !Jitter_num_svm[1] && !Jitter_num_svm[2]))
	{
		crops.resize(1);
		dlib::assign_image(crops[0], img); //dest, src
	}
	return crops;
}

std::vector<dlib::matrix<float, 0, 1>> NullImplFRVT11::array_to_dlib_1D_matrix(int face_count, float* in_array, int dim_size) {
	std::vector<dlib::matrix<float, 0, 1>> out_matrix(face_count);
	for (int i = 0; i < face_count; i++) {
		out_matrix[i].set_size(dim_size);
		for (int j = 0; j < dim_size; j++)
			out_matrix[i](j) = in_array[i*dim_size + j];
	}
	return out_matrix;
}

std::string NullImplFRVT11::ProduceUUID(){
    std::srand(time(NULL));
    char strUuid[128];
    sprintf(strUuid, "%x%x-%x-%x-%x-%x%x%x", 
    rand(), rand(),                 // Generates a 64-bit Hex number
    rand(),                         // Generates a 32-bit Hex number
    ((rand() & 0x0fff) | 0x4000),   // Generates a 32-bit Hex number of the form 4xxx (4 indicates the UUID version)
    rand() % 0x3fff + 0x8000,       // Generates a 32-bit Hex number in the range [0x8000, 0xbfff]
    rand(), rand(), rand());        // Generates a 96-bit Hex number
    std::string outputString = strUuid;
    return outputString;
}


void NullImplFRVT11::Deallocator(void* data, size_t length, void* arg)
{
	std::free(data);
}