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
    //For FD
    face_input_detector = dlib::get_frontal_face_detector();

    if(input_image){
        delete[] input_image;
        input_image = NULL;
    }
}

ReturnStatus
NullImplFRVT11::initialize(const std::string &configDir)
{
	try { 
        face_input_detector = dlib::get_frontal_face_detector();
        std::string LandMarkFileName = configDir + "/geo_vision_5_face_landmarks.dat";
        dlib::deserialize(LandMarkFileName) >> sp_5; //read dlib landmark model
        //tbb::global_control(tbb::global_control::max_allowed_parallelism, 1);
        // imgCount = 0;
        // detectFailCount = 0;
        //tbbControl = new tbb::global_control(tbb::global_control::max_allowed_parallelism, 1);
        m_JitterCount = FR_JITTER_COUNT;
        if(!input_image){
            input_image = new unsigned char [FR_IMAGE_HEIGHT * FR_IMAGE_HEIGHT *3];
        }	        
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
        // clock_t begin = clock();
        for (unsigned int i=0; i<faces.size(); i++) {
            mtx.lock();
            // cout << "00" << endl;;
            // imgCount++;
            // ----------------------------------------------------------------------------------------------------
            // std::list<Face::Ptr> facesAttributes;
            size_t id = 0;
            saveImgMtx.lock();
            cv::Mat frame = cv::Mat(faces[i].height, faces[i].width, CV_8UC3);
            cv::Mat showframe;
            // -------------------------------Set input data----------------------------------------------------
            // slog::info << "frvt imput image height: " << faces[i].height << ", width: " << faces[i].width << ", size: " << faces[i].size() << slog::endl;
            std::memcpy(frame.data, faces[i].data.get(), faces[i].size() );  
            cv::cvtColor(frame,frame, cv::COLOR_BGR2RGB);
            frame.copyTo(showframe);
            string chipFileName = "FDresults/OriImg(" + ProduceUUID() + ").jpg";
            cv::imwrite(chipFileName, frame);
            saveImgMtx.unlock();
            dlib::matrix<dlib::bgr_pixel> enroll_chip; //original extract chip
            std::vector<dlib::point> parts;
            dlib::cv_image<dlib::bgr_pixel> cv_imgFR(frame);
            dlib::matrix<dlib::bgr_pixel> imgFR;
            assign_image(imgFR, cv_imgFR);
            std::vector<dlib::rectangle> face_det = face_input_detector(imgFR);
            // For multi detected face
            int maxFaceId = 0;
            int maxRectArea = 0;
            if(face_det.size() > 0){
                // cout << "031" << endl;
                    if(face_det.size() > 1){
                    for (size_t j = 0; j < face_det.size(); j++) {
                        if(face_det[j].width() * face_det[j].height() >  maxRectArea){
                            maxRectArea = face_det[j].width() * face_det[j].height();
                            maxFaceId = j;
                        }
                    }
                }
                //====================Do dlib Landmark====================================
                dlib::full_object_detection shape_5 = sp_5(imgFR, face_det[0]);
                cv::Point pt1(face_det[maxFaceId].left(), face_det[maxFaceId].top());
                // and its bottom right corner.
                cv::Point pt2(face_det[maxFaceId].right(), face_det[maxFaceId].bottom());
                // These two calls...
                cv::rectangle(showframe, pt1, pt2, cv::Scalar(0, 0, 255));

                // --------------------------- Assign Landmark for eye center----------------------------
                //dlibLandmark leftEye:2 3 rightEye:1 0 nosePhiltrum:4
                int xleftEyeCenter = int ((shape_5.part(2).x() + shape_5.part(3).x())*0.5);
                int yleftEyeCenter = int ((shape_5.part(2).y() + shape_5.part(3).y())*0.5);
                cv::circle(showframe, cv::Point(xleftEyeCenter, yleftEyeCenter), 1 + static_cast<int>(0.012 * face_det[maxFaceId].width()), cv::Scalar(255, 0, 0), -1);
                int xRightEyeCenter = int ((shape_5.part(0).x() + shape_5.part(1).x())*0.5);
                int yRightEyeCenter = int ((shape_5.part(0).y() + shape_5.part(1).y())*0.5);
                cv::circle(showframe, cv::Point(xRightEyeCenter, yRightEyeCenter), 1 + static_cast<int>(0.012 * face_det[maxFaceId].width()), cv::Scalar(0, 0, 255), -1);
                eyeCoordinates.clear();
                eyeCoordinates.shrink_to_fit();
                eyeCoordinates.push_back(EyePair(true, true, xRightEyeCenter, yRightEyeCenter, xleftEyeCenter, yleftEyeCenter));
                //////////ISO standard: The label "left" refers to subject's left eye (and similarly for the right eye), such that xright < xleft/////////////////
                // cout << "eyeCoordinatesLeftEye("<< i << "): (x,y)=(" << eyeCoordinates[i].xleft << "," << eyeCoordinates[i].yleft << ")"  << endl;
                // cout << "eyeCoordinatesRightEye("<< i << "): (x,y)=(" << eyeCoordinates[i].xright << "," << eyeCoordinates[i].yright << ")"  << endl;
                saveImgMtx.lock();
                string detectFileName = "FDresults/face(" + ProduceUUID() + ").jpg";
                dlib::cv_image<dlib::bgr_pixel> cv_temp(showframe);
                dlib::matrix<dlib::bgr_pixel> dlib_array2d;
                dlib::assign_image(dlib_array2d, cv_temp);
                dlib::save_jpeg(dlib_array2d,detectFileName,100);
                // cv::imwrite(detectFileName, showframe);
                saveImgMtx.unlock();
                // ---------------------------------------------------------------------------------------
                
                //dlib::rectangle known_det;
                // dlib::matrix<dlib::bgr_pixel> enroll_chip; //original extract chip
                // // known_det.set_left(rect.x);
                // // known_det.set_top(rect.y);
                // // known_det.set_right(rect.x + rect.width);
                // // known_det.set_bottom(rect.y + rect.height);
                // // slog::info << "known_det("<<  known_det.left() << "," << known_det.right() << "," <<
                // //                  known_det.top() << "," << known_det.bottom() << ")" << slog::endl;
                // std::vector<dlib::point> parts;
                // parts.resize(5);
                // //mapping to dlibLandmark leftEye:2 3 rightEye:1 0 nosePhiltrum:4
                // parts[0].x() = rect.x + rect.width * (normed_landmarks[2 * 3]);
                // parts[0].y() = rect.y + rect.height * (normed_landmarks[2 * 3 + 1]);
                // parts[1].x() = rect.x + rect.width * (normed_landmarks[2 * 2]); 
                // parts[1].y() = rect.y + rect.height * (normed_landmarks[2 * 2 + 1]);
                // parts[2].x() = rect.x + rect.width * (normed_landmarks[2 * 1]); 
                // parts[2].y() = rect.y + rect.height * (normed_landmarks[2 * 1 + 1]);
                // parts[3].x() = rect.x + rect.width * (normed_landmarks[2 * 0]); 
                // parts[3].y() = rect.y + rect.height * (normed_landmarks[2 * 0 + 1]);
                // parts[4].x() = rect.x + rect.width * (normed_landmarks[2 * 5]); 
                // parts[4].y() = rect.y + rect.height * (normed_landmarks[2 * 5 + 1]);
                // for (int k = 0; k < 5; k++) {
                //     cv::circle(showframe, cv::Point(parts[k].x(),  parts[k].y()), 1 + static_cast<int>(0.012 * rect.width), cv::Scalar(0, 255, 255), -1);
                //     string lmText = to_string(k); 
                //     cv::putText(showframe, lmText, cv::Point(parts[k].x(),  parts[k].y()), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0));
                // }
                // cv::Point pt1(rect.x, rect.y);
                // // and its bottom right corner.
                // cv::Point pt2(rect.x + rect.width, rect.y + rect.height);
                // // These two calls...
                // cv::rectangle(showframe, pt1, pt2, cv::Scalar(0, 0, 255));
                // cv::imshow("Detection results", showframe);
                // saveImgMtx.lock();
                // std::time_t t = std::time(0);   // get time now
                // std::tm* now = std::localtime(&t);
                // srand((unsigned) time(&t));
                // int rndNumber = rand() % 10000;
                // string detectFileName = "FDresults/face(" + to_string(now->tm_year + 1900) + "_"
                // + to_string(now->tm_mon + 1) + "_"  + to_string(now->tm_mday) + "_" + to_string(now->tm_hour) + "_" 
                // + to_string(now->tm_min) + "_" + to_string(now->tm_sec) + "_" + to_string(rndNumber) + ").jpg"; 
                // string detectFileName = "FDresults/face(" + ProduceUUID() + ").jpg"; ;
                // cv::imwrite(detectFileName, showframe);
                // saveImgMtx.unlock();
                // cv::waitKey(300);
                // cv::destroyAllWindows();
                // cout << "01" << endl;
                // ---------------------------------------------------------------------------------------



                //-------------------------------Crop face image================================
                dlib::extract_image_chip(imgFR, dlib::get_face_chip_details(shape_5, FR_IMAGE_HEIGHT, FR_IMAGE_PADDING*0.01), enroll_chip);
                // cv::Mat enrollChipMat = dlib::toMat(enroll_chip);
                saveImgMtx.lock();
                std::time_t t = std::time(0);   // get time now
                std::tm* now = std::localtime(&t);
                srand((unsigned) time(&t));
                int rndNumber = rand() % 10000;
                string chipFileName = "FDresults/chip(" + ProduceUUID() + ").jpg";
                dlib::save_jpeg(enroll_chip,chipFileName,100);
                // cv::imwrite(chipFileName, enrollChipMat);
                saveImgMtx.unlock();
                // cv::waitKey(300);
                // cv::destroyAllWindows();
                // cout << "06" << endl;
                
            }else{//no face detected return error enum
                //return ReturnStatus(ReturnCode::FaceDetectionError);
            }            
            
            // std::vector<dlib::matrix<float, 0, 1>> SVM_descriptor;
            // std::vector<dlib::matrix<dlib::bgr_pixel>> SVM_distrub_color_crops;
            // int cropsCount = m_JitterCount;
            // if(role == TemplateRole::Enrollment_11 || role == TemplateRole::Enrollment_1N){
            //     // slog::info << "FR image TemplateRole Enrollment"<< slog::endl;
            //     if(m_JitterCount > 0){
            //         SVM_distrub_color_crops = this->jitter_image(enroll_chip, FR_IMAGE_HEIGHT, FR_IMAGE_HEIGHT);
            //     }else{
            //         SVM_distrub_color_crops.push_back(enroll_chip);
            //         cropsCount = 1;
            //     }
            // }else{
            //     // slog::info << "FR image TemplateRole Verification"<< slog::endl;
            //     SVM_distrub_color_crops.push_back(enroll_chip);
            //     cropsCount = 1;
            // }

            // cout << "07" << endl;
            

            // for (int i = 0; i < cropsCount; i++)
            // {
            //     cv::Mat chipMat = dlib::toMat(SVM_distrub_color_crops[i]);
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
            //         memset(jitterFR_emb,0.0,FR_EMBEDDING_SIZE);
            //         memset(gender,0.0,2);
            //         memset(age,0.0,7);
            //         for (int out_c = 0; out_c < network_OutputName.size(); out_c++) {
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
            //         SVM_descriptor.push_back(array_to_dlib_1D_matrix(1, jitterFR_emb, FR_EMBEDDING_SIZE)[0]);
            //         // FR_emb[emb]
            //     } //jitter cropsCount
            //     // -----------------------------------------------------------------------------------------------------
            //     // slog::info << "FR features[0,1,127,510,511]: " 
            //     // << "[" << FR_emb[0] << ", " << FR_emb[1] << ", " << FR_emb[127] << ", " << FR_emb[510] << ", " << FR_emb[511] << "] " << slog::endl;
            //     dlib::matrix<float, 0, 1> temp_mat = mean(mat(SVM_descriptor));
            //     std::vector<dlib::matrix<float, 0, 1>> EnrollDescriptor;
            //     int normalizeLength = dlib::length(temp_mat) < 1 ? 1 : dlib::length(temp_mat);
            //     EnrollDescriptor.push_back(temp_mat / normalizeLength); //Use jitter image and normalize to length
            //     memset(FR_emb,0.0,FR_EMBEDDING_SIZE);
            //     for (int j = 0; j < FR_EMBEDDING_SIZE; j++)
            //     {
            //         FR_emb[j] = EnrollDescriptor[EnrollDescriptor.size() - 1](j, 0);
            //     }
            //     std::vector <dlib::matrix<float, 0, 1>>().swap(EnrollDescriptor);
            // } //detected faces vector array

            std::vector<float> fv;
            if(face_det.size() == 0){ //for no FD found give false eyes detected bool and zero coordinates
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
            memset(FR_emb,0,__SIZEOF_FLOAT__*FR_EMBEDDING_SIZE);
            for(int emb = 0; emb < FR_EMBEDDING_SIZE; emb++){
                fv.push_back(FR_emb[emb]);
            }
            //==test use==
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(fv.data());
            int dataSize = sizeof(float) * fv.size();
            templ.resize(dataSize);
            std::memcpy(templ.data(), bytes, dataSize);
            // slog::info << "FR features size: "<<fv.size()<< " fv[0,1,127,510,511]: " 
            // << "[" << fv[0] << ", " << fv[1] << ", " << fv[127] << ", " << fv[510] << ", " << fv[511] << "] " << slog::endl;
            mtx.unlock();

                // cout << "08" << endl;
                

        } //faces vect`or size array
        // clock_t end = clock();
        // double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        // slog::info << "FR createTemplate executeㄥtime: "<<time_spent<< " sec spent" << slog::endl;
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
    }
    return ReturnStatus(ReturnCode::Success);
}

ReturnStatus
NullImplFRVT11::matchTemplates(
        const std::vector<uint8_t> &verifTemplate,
        const std::vector<uint8_t> &enrollTemplate,
        double &similarity)
{
    float *featureVector = (float *)enrollTemplate.data();
    float *vfeatureVector = (float *)verifTemplate.data();
    dlib::matrix<float, 0, 1> out_matrix;
    dlib::matrix<float, 0, 1> vout_matrix;
    out_matrix.set_size(FR_EMBEDDING_SIZE);
    vout_matrix.set_size(FR_EMBEDDING_SIZE);
    for (int j = 0; j < FR_EMBEDDING_SIZE; j++){
        out_matrix(j) = featureVector[j];
        vout_matrix(j) = vfeatureVector[j];
    }
    similarity = 1.00 - (dlib::length(out_matrix - vout_matrix)*0.50 - 0.20);
    // slog::info << "out_matrix[0,1,127,510,511]: " 
    // << "[" << out_matrix(0) << ", " << out_matrix(1) << ", " << out_matrix(127) << ", " << out_matrix(510) << ", " << out_matrix(511) << "] " << slog::endl;
    // slog::info << "vout_matrix[0,1,127,510,511]: " 
    // << "[" << vout_matrix(0) << ", " << vout_matrix(1) << ", " << vout_matrix(127) << ", " << vout_matrix(510) << ", " << vout_matrix(511) << "] " << slog::endl;
    // slog::info << "similarity: " << similarity << slog::endl;
    // similarity = rand() % 1000 + 1;
    return ReturnStatus(ReturnCode::Success);
}

std::shared_ptr<Interface>
Interface::getImplementation()
{
    return std::make_shared<NullImplFRVT11>();
}

std::vector<dlib::matrix<dlib::bgr_pixel>> NullImplFRVT11::jitter_image(const dlib::matrix<dlib::bgr_pixel>& img, int height, int width)
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
	std::vector <dlib::matrix<dlib::bgr_pixel>>().swap(crops);

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
			dlib::matrix<dlib::bgr_pixel> temp;
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
    srand(time(NULL));
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