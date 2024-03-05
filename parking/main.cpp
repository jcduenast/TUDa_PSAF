// C++ program for the above approach 
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <cmath>
  
// Library to include for 
// drawing shapes
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>      // contours
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>

// para el manejo de strings
#include <sstream>
#include <iomanip>

#include <opencv2/ximgproc.hpp>

// for blobs
#include <opencv2/features2d.hpp>

#include "main.hpp"

void parking_detection_test(int set, int expSet);
void test_img_stitching();
cv::Mat parking_detection(cv::Mat img);
cv::Mat get_eagle_view(cv::Mat img_in);
cv::Mat imgStitching(cv::Mat img1, cv::Mat img2);
cv::Mat projectOnRaw(cv::Mat raw, cv::Mat classified);

int main (int argc, char *argv[]){
    int set = 2;
    int expSet = 3;    // 0 para run, 1 para lisiado, 2 para tracking, 3 invalido
    if(argc > 1){
        set = atoi(argv[1]);
        if(argc > 2){
            expSet = atoi(argv[2]);
        }
    }

    parking_detection_test(set, expSet);
    // test_img_stitching();
    

    return 0;
}

void test_img_stitching(){
    cv::Mat result, raw1, raw2;
    int img_id = 165;
    raw1 = cv::imread("/home/ubi/usb/final_este_si02/raw_img_167.jpg");
    raw2 = cv::imread("/home/ubi/usb/final_este_si02/raw_img_169.jpg");
    result = imgStitching(raw1, raw2);
    cv::imshow("result", result);
    cv::waitKey(0);
    return;
}

void parking_detection_test(int set, int expSet){
    int frame = 0;
    std::string root_path, cam_img_name;
    cv::Mat og_img, parking_img;

    std::string local_root_path = "/home/ubi/usb/";  // pa' camilo
    // std::string local_root_path = "/home/daniel/Documentos/TU/PSAF/TUDa_PSAF/camera_processing/";  // pa' Daniel

    
    std::stringstream ss;
    ss << std::setw(2) << std::setfill('0') << set;
    std::string run_id_string = ss.str();
    root_path = local_root_path + "invalido" + run_id_string + "/";
    
    // root_path = "/home/ubi/usb/run" + run_id_string + "/";
    // root_path = "/home/ubi/TUDa_PSAF/camera_processing/test/"; // path for camilo
    // root_path = "/home/daniel/Documentos/TU/PSAF/TUDa_PSAF/camera_processing/test/"; // path for Daniel
    cv::Mat raw_info;

    for(;; frame++){
        cam_img_name = root_path + "raw_img_" + std::to_string(frame) + ".jpg";        // hasta la 2 está con png, de ahí en adelante con .jpg
        std::cout << "Frame: " << std::to_string(frame) << " at: " << cam_img_name << std::endl;
        og_img = cv::imread(cam_img_name);
        parking_img = parking_detection(og_img);
        raw_info = projectOnRaw(og_img, parking_img);
        cv::imshow("Parking detection", parking_img);
        cv::imwrite(root_path+"parking_detection_"+std::to_string(frame) + ".jpg", parking_img);
        cv::imshow("Parking detection on raw", raw_info);
        cv::imwrite(root_path+"parking_detection_on_raw_"+std::to_string(frame) + ".jpg", raw_info);
        cv::waitKey(30);
    }
    return;
}

cv::Mat imgStitching(cv::Mat img1, cv::Mat img2){
    cv::Mat result, eagle1, eagle2;
    eagle1 = get_eagle_view(img1);
    eagle2 = get_eagle_view(img2);

    bool divide_images = false;
    // cv::Stitcher::Mode mode = cv::Stitcher::PANORAMA;
    cv::Stitcher::Mode mode = cv::Stitcher::SCANS;
    std::vector<cv::Mat> imgs={eagle1, eagle2};
    std::string result_name = "result.jpg";
    
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(mode);
    cv::Stitcher::Status status = stitcher->stitch(imgs, result);

    // cv::imwrite(result_name, result);


    return result;
}

cv::Mat parking_detection(cv::Mat img){

    cv::Mat classified = cv::Mat().zeros(img.size(), CV_8UC3);
    cv::Mat gray, blurred, binary, binary_eagle;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 0, 0, cv::BORDER_DEFAULT);
    int block_size=101, const_subtrahend=-50;
    cv::adaptiveThreshold(blurred, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block_size, const_subtrahend);
    binary_eagle = get_eagle_view(binary);
    
    // experiment
    // cv::Mat test = cv::Mat(); //.zeros(img.size(), CV_8UC3);
    // cv::Mat test2 = cv::Mat(); //.zeros(img.size(), CV_8UC3);
    // cv::Mat gray2, blurred2, binary2, eagle;
    // eagle = get_eagle_view(img);
    // cv::cvtColor(eagle, gray2, cv::COLOR_BGR2GRAY);
    // cv::GaussianBlur(gray2, blurred2, cv::Size(9, 9), 0, 0, cv::BORDER_DEFAULT);
    // cv::adaptiveThreshold(blurred2, test, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block_size, const_subtrahend);
    // binary_eagle = blurred.clone();
    // cv::imshow("blurred", blurred2);

    // cv::GaussianBlur(binary_eagle, test2, cv::Size(9, 9), 0, 0, cv::BORDER_DEFAULT);
    // cv::adaptiveThreshold(test2, test2, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block_size, const_subtrahend);
    // cv::imshow("test2", test2);

    std::vector<std::vector<cv::Point>> cnt;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary_eagle, cnt, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE); // usefull for finding the parking spot
    // cv::findContours(test2, cnt, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE); // usefull for finding the parking spot
    // cv::findContours(binary_eagle, cntAll, hierarchyAll, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE); // gets only the external contours

    std::vector<bool> isPerpendicularParking(cnt.size(), false);
    std::vector<cv::Rect> boundRect(cnt.size());
    std::vector<cv::RotatedRect> boundMinArea(cnt.size());
    cv::Point2f rotatedRectPoints_aux[4];
    std::vector<cv::Vec4f> lineCnt(cnt.size());

    int cntBiggestIdx = -1;
    size_t cntBiggestSize = 0;

    cv::Point centerParkingSpot;

    for (int i=0; i<cnt.size(); i++){
        if (cnt.at(i).size() > 300 && cnt.at(i).size() < 450 && hierarchy.at(i)[3]!=-1)   isPerpendicularParking.at(i) = true;
        cv::drawContours(classified, cnt, i, cv::Scalar(100,100,100), cv::FILLED, cv::LINE_8);
    }

    for (size_t i=0; i<cnt.size(); i++){
        boundRect[i] = cv::boundingRect(cnt[i]);
        boundMinArea[i] = cv::minAreaRect(cnt[i]);
                
        if (isPerpendicularParking.at(i)){
            cv::rectangle(classified, boundRect[i].tl(), boundRect[i].br(), cv::Scalar(150,150,150), 1, cv::LINE_8);
            cv::drawContours(classified, cnt, i, cv::Scalar(255,0,0), cv::FILLED, cv::LINE_8);
            boundMinArea[i].points(rotatedRectPoints_aux);
            for (int j=0; j<4; j++) cv::line(classified, rotatedRectPoints_aux[j], rotatedRectPoints_aux[(j+1)%4], cv::Scalar(255,255,255));
            cv::putText(classified, std::to_string(cnt.at(i).size()), cv::Point(boundRect[i].x, boundRect[i].y+15), cv::FONT_HERSHEY_COMPLEX_SMALL , 0.8, CV_RGB(255,255,255), 1, cv::LINE_8, false);
            centerParkingSpot = cv::Point(boundRect[i].x + boundRect[i].width / 2, boundRect[i].y + boundRect[i].height / 2);
            cv::circle(classified, centerParkingSpot, 5, cv::Scalar(0, 255, 0), 2);
        }
    }
    // cv::imshow("Test", test);
    return classified;
}

cv::Mat get_eagle_view(cv::Mat img_in){
    cv::Mat projected;
    cv::Mat homography;
    homography = (cv::Mat(3, 3, CV_64F, homography_data_juan_daniel_hd)).clone();
    // cv::warpPerspective(img_in, projected, homography, cv::Size(640, 490), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    cv::warpPerspective(img_in, projected, homography, cv::Size(640, 490));
    return projected;
}

cv::Mat projectOnRaw(cv::Mat raw, cv::Mat classified){
    cv::Mat raw_eagle, raw_eagle_bg, homography, raw_superposed_roi, raw_superposed_roi_eagle, class_mask, class_mask_inv, class_gray, class_fg;
    homography = (cv::Mat(3, 3, CV_64F, homography_data_juan_daniel_hd)).clone();
    cv::warpPerspective(raw, raw_eagle, homography, cv::Size(640, 480));            // project to eagle perspective
    cv::cvtColor(classified, class_gray, cv::COLOR_BGR2GRAY);                       // color to grayscale
    cv::threshold(class_gray, class_mask_inv, 1, 255, cv::THRESH_BINARY_INV);       // take regions in black (inverse of low threshold)
    cv::bitwise_and(raw_eagle, raw_eagle, raw_eagle_bg, class_mask_inv);            // black out sections of the eagle raw to overwrite
    cv::bitwise_not(class_mask_inv, class_mask);                                    // info on white
    cv::bitwise_and(classified, classified, class_fg, class_mask);                  // blackout classifier parts not to pass along
    cv::add(raw_eagle_bg, class_fg, raw_superposed_roi_eagle);                      // add background and foreground
    cv::warpPerspective(raw_superposed_roi_eagle, raw_superposed_roi, homography, cv::Size(640, 480), cv::WARP_INVERSE_MAP);
    cv::Mat raw_roi, raw_roi_mask, raw_roi_mask_inv, raw_bg, raw_superposed_roi_gray, raw_superposed;
    cv::cvtColor(raw_superposed_roi, raw_superposed_roi_gray, cv::COLOR_BGR2GRAY);  // get gray from color
    cv::threshold(raw_superposed_roi_gray, raw_roi_mask_inv, 1, 255, cv::THRESH_BINARY_INV);    // take regions in black
    cv::bitwise_and(raw, raw, raw_bg, raw_roi_mask_inv);                            // create background
    cv::add(raw_bg, raw_superposed_roi, raw_superposed);
    return raw_superposed;
}
