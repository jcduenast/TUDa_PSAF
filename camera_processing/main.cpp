// C++ program for the above approach 
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <cmath>
  
// Library to include for 
// drawing shapes 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <opencv2/opencv.hpp>

// para el manejo de strings
#include <sstream>
#include <iomanip>

#include <opencv2/ximgproc.hpp>

#include "main.hpp"

// using namespace cv;
// using namespace std;

void setup_test();
void compare_record_w_own();
cv::Mat get_eagle_view(cv::Mat img_in, int mode);
cv::Mat inf_processing(cv::Mat camera_raw_color, int mode);
cv::Mat car_on_projection(cv::Mat img_in, int mode);
cv::Mat final_on_og(cv::Mat img_final, cv::Mat img_og); // img_final U8C1
std::vector<std::vector<cv::Point>> proc_proposal(cv::Mat camera_raw_color);
void test_algo(int mode, int set);

int main (){
    // setup_test();
    // compare_record_w_own();
    test_algo(2, 7);
    return 0;
}

void test_algo(int mode, int set){
    int frame = 0;
    std::string root_path, cam_img_name, prc_img_name;
    cv::Mat own_processed, car_processed, og_img, eagle_view_color, own_processed_overlay;

    std::stringstream ss;
    ss << std::setw(2) << std::setfill('0') << set;
    std::string run_id_string = ss.str();
    root_path = "/home/ubi/usb/run" + run_id_string + "/";

    for(;; frame++){
        cam_img_name = root_path + "raw_img_" + std::to_string(frame) + ".jpg";        // hasta la 2 está con png, de ahí en adelante con .jpg
        og_img = cv::imread(cam_img_name);                                              // cargar la imagen de la camara a color
        proc_proposal(og_img);
        // eagle_view_color = get_eagle_view(og_img, mode);                                // eagle view de la imagen original, a color
        // own_processed = inf_processing(og_img, mode);                                   // imagen raw a color procesada por los infos
        // own_processed_overlay = final_on_og(own_processed, eagle_view_color);
        // cv::imshow("Own processing overlayed on color", own_processed_overlay);
        // cv::waitKey(0);
    }
    return;
}

std::vector<std::vector<cv::Point>> proc_proposal(cv::Mat camera_raw_color){
    std::vector<std::vector<cv::Point>> lines;
    int block_size=101, const_subtrahend=-50, mode=2;
    cv::Mat gray, binary, binary_eagle, blurred, color_eagle_std, color_eagle_plt;
    cv::cvtColor(camera_raw_color, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 0, 0, cv::BORDER_DEFAULT);
    cv::adaptiveThreshold(blurred, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block_size, const_subtrahend);
    binary_eagle = get_eagle_view(binary, mode);
    color_eagle_std = get_eagle_view(camera_raw_color, mode);
    color_eagle_plt = get_eagle_view(camera_raw_color, mode);

    cv::Mat canny_dst, thinned;
    cv::Canny(binary_eagle, canny_dst, 50, 200, 3);
    
    
    cv::Mat algo_input = canny_dst.clone();
    cv::imshow("Algo input", algo_input);
    // std Hough
    std::vector<cv::Vec2f> std_lines;
    cv::HoughLines(canny_dst, std_lines, 1, CV_PI/180, 150, 0, 0);
    // draw std hough lines
    for(size_t i=0; i<std_lines.size(); i++){
        float rho = std_lines[i][0], theta = std_lines[i][1];
        cv::Point pt1, pt2;
        double a=cos(theta), b=sin(theta);
        double x0=a*rho, y0=b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        cv::line(color_eagle_std, pt1, pt2, cv::Scalar(0,0,255), 1, cv::LINE_AA);
    }

    // probabilistic line transform plt
    std::vector<cv::Vec4i> plt_lines;
    cv::HoughLinesP(canny_dst, plt_lines, 1, CV_PI/180, 50, 50, 10);
    // draw plt lines
    for(size_t i=0; i<plt_lines.size(); i++){
        cv::Vec4i l = plt_lines[i];
        cv::line(color_eagle_plt, cv::Point(l[0],l[1]), cv::Point(l[2],l[3]), cv::Scalar(0,255,0), 1, cv::LINE_AA);
    }

    cv::imshow("Std hough", color_eagle_std);
    cv::imshow("Plt hough", color_eagle_plt);
    cv::waitKey(0);
    return lines;
}

void compare_record_w_own(){
    int mode = 2;   // 0 old, 1 new calib, 2 new calib wider
    int id = 0;     // frame to be processed
    int run = 7;
    std::string root_path, cam_img_name, prc_img_name;
    cv::Mat own_processed, car_processed, og_img, eagle_view_color, own_processed_overlay;

    std::stringstream ss;
    ss << std::setw(2) << std::setfill('0') << run;
    std::string run_id_string = ss.str();
    root_path = "/home/ubi/usb/run" + run_id_string + "/";  
    for(;; id++){
        cam_img_name = root_path + "raw_img_" + std::to_string(id) + ".jpg";        // hasta la 2 está con png, de ahí en adelante con .jpg
        prc_img_name = root_path + "processed_img_" + std::to_string(id) + ".jpg";  // imagen tomada desde el carrito y procesada por los infos
        car_processed = cv::imread(prc_img_name);                                   // cargar la imagen ya procesada
        og_img = cv::imread(cam_img_name);                                          // cargar la imagen de la camara a color
        eagle_view_color = get_eagle_view(og_img, mode);                            // eagle view de la imagen original, a color
        own_processed = inf_processing(og_img, mode);                               // imagen raw a color procesada por los infos
        // cv::imshow("Own processing", own_processed);
        cv::imshow("Processed in the car", car_processed);
        own_processed_overlay = final_on_og(own_processed, eagle_view_color);
        cv::imshow("Own processing on color", own_processed_overlay);
        cv::waitKey(0);
    }
    return;
}

void setup_test(){
    int mode = 0; // 0 old, 1 new calib, 2 fav wider
    std::string file_name = "/home/ubi/usb/run07/raw_img_1.jpg";
    cv::Mat raw_color_img = cv::imread(file_name);

    cv::Mat img_out, img_gray, gray, binary, transformed, blurred;
    cv::cvtColor(raw_color_img, gray, cv::COLOR_BGR2GRAY);
    cv::imshow("Gray", gray);
    cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 0, 0, cv::BORDER_DEFAULT);
    cv::imshow("Blurred", blurred);
    // cv::adaptiveThreshold(blurred, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block_size, const_subtrahend);
    cv::threshold(blurred, binary, 155, 255, cv::THRESH_BINARY);
    cv::imshow("Binary", binary);
    transformed = get_eagle_view(binary, mode);
    cv::imshow("Transformed", transformed);
    cv::waitKey(0);

    cv::Mat img_eagle_view = get_eagle_view(raw_color_img, mode);
    cv::Mat processed = inf_processing(raw_color_img, mode);
    cv::Mat processed_C3;
    std::vector<cv::Mat> copies{processed,processed,processed};
    cv::merge(copies,processed_C3);
    
    cv::Mat processed_w_car = car_on_projection(processed_C3, mode);
    cv::imshow("Test processing with car", processed_w_car);
    
    cv::Mat img_w_car = car_on_projection(img_eagle_view, mode);
    cv::imshow("Test img with car", img_w_car);
    
    cv::Mat comparison = final_on_og(processed, img_eagle_view);
    cv::imshow("Processed on og eagle", comparison);
    cv::waitKey(0);
}

cv::Mat final_on_og(cv::Mat img_final, cv::Mat img_og){
    cv::Mat output;
    cv::Mat final_not;
    cv::bitwise_not(img_final, final_not);
    img_og.copyTo(output, final_not);
    return output;
}

cv::Mat inf_processing(cv::Mat camera_raw_color, int mode){
    int block_size = 101;
    int const_subtrahend = -50;
    cv::Mat img_out, img_gray, gray, binary, transformed, blurred;
    cv::cvtColor(camera_raw_color, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 0, 0, cv::BORDER_DEFAULT);
    cv::adaptiveThreshold(blurred, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block_size, const_subtrahend);
    // cv::threshold(blurred, binary, 155, 255, cv::THRESH_BINARY);
    transformed = get_eagle_view(binary, mode);
    return transformed;
}

cv::Mat get_eagle_view(cv::Mat img_in, int mode){
    cv::Mat projected;
    cv::Mat homography;
    switch (mode){
        case 0:     // las que usaban ellos hasta ahora
            homography = (cv::Mat(3, 3, CV_64F, homography_data_psaf1000_old)).clone();
            cv::warpPerspective(img_in, projected, homography, cv::Size(640, 640));
        break;
        case 1:
            homography = (cv::Mat(3, 3, CV_64F, homography_data_psaf1000)).clone();
            cv::warpPerspective(img_in, projected, homography, cv::Size(480, 720));
        break;
        case 2:     // la que más me gusta :v
            homography = (cv::Mat(3, 3, CV_64F, homography_data_psaf1000_wider)).clone();
            cv::warpPerspective(img_in, projected, homography, cv::Size(640, 640));
        break;
        default:
            homography = (cv::Mat(3, 3, CV_64F, homography_data_psaf1000_wider)).clone();
            cv::warpPerspective(img_in, projected, homography, cv::Size(640, 640));
            break;
    }
    return projected;
}

cv::Mat car_on_projection(cv::Mat img_in, int mode){
    cv::Mat img_w_car;
    int roi_width, roi_height;
    int car_width, car_height;

    switch (mode){      // ancho carrito 155mm | largo carrito 258mm
        case 0:
            // old param: ego_scale_mm2px: 0.6 | 300 mm hacia abajo -> 180 px
            roi_width = 640;
            roi_height = 900;
            car_width = 93;
            car_height = 155;
        break;
        case 1:
            // ego_scale_mm2px: 0.6 | el ancho que debería ser
            roi_width = 480;
            roi_height = 900;
            car_width = 93;
            car_height = 155;
        break;
        case 2:
            // top wide: ego_scale_mm2px: 0.5333333333333333
            roi_width = 640;
            roi_height = 800;
            car_width = 83;
            car_height = 134;
        break;
        default:
            roi_width = 640;
            roi_height = 800;
            car_width = 83;
            car_height = 134;
        break;
    }
    
    img_w_car = cv::Mat::zeros(cv::Size(roi_width,roi_height), CV_8UC3);
	cv::Point pt1(roi_width/2-car_width/2, roi_height-car_height), pt2(roi_width/2+car_width/2, roi_height);
	cv::rectangle(img_w_car, pt1, pt2, cv::Scalar(150, 150, 150));
	cv::arrowedLine(img_w_car, cv::Point(roi_width/2,roi_height), cv::Point(roi_width/2, roi_height-50), cv::Scalar(0,0,255), 4);
	cv::arrowedLine(img_w_car, cv::Point(roi_width/2,roi_height), cv::Point(roi_width/2-50, roi_height), cv::Scalar(0,255,0), 4);
    img_in.copyTo(img_w_car(cv::Rect(0,0,img_in.cols, img_in.rows)));
    return img_w_car;
}
