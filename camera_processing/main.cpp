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
void drawHoughStd(cv::Mat canvas, std::vector<cv::Vec2f> std_lines, cv::Scalar color, int thickness);
void drawHoughPlt(cv::Mat canvas, std::vector<cv::Vec4i> plt_lines, cv::Scalar color, int thickness);
int max(int num1, int num2);
int min(int num1, int num2);

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
    root_path = "/home/daniel/Documentos/TU/PSAF/TUDa_PSAF/camera_processing/test/"; //+ run_id_string + "/";

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
    // // Preprocessing
    std::vector<std::vector<cv::Point>> lines;  // output
    cv::Mat gray, blurred, binary, binary_eagle;
    cv::cvtColor(camera_raw_color, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 0, 0, cv::BORDER_DEFAULT);
    int block_size=101, const_subtrahend=-50;
    cv::adaptiveThreshold(blurred, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block_size, const_subtrahend);
    int mode=2; // 2 para la transformación más amplia
    binary_eagle = get_eagle_view(binary, mode);

    // // closing operation
    int morph_size = 2; 
    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size)); 
    cv::Mat binary_closed;
    cv::morphologyEx(binary_eagle, binary_closed, cv::MORPH_CLOSE, element, cv::Point(-1, -1), 2);
    
    cv::Mat color_eagle = get_eagle_view(camera_raw_color, mode);

    // with binary only: -----------------------------------------------------------------------------------------
    cv::Mat color_binary_output = color_eagle.clone();
    
    std::vector<cv::Vec4i> plt_binary_lines;
    cv::HoughLinesP(binary_eagle, plt_binary_lines, 1, CV_PI/180, 50, 50, 50);
    drawHoughPlt(color_binary_output, plt_binary_lines, cv::Scalar(0,255,0), 2);
    // // std Hough
    // std::vector<cv::Vec2f> std_binary_lines;
    // cv::HoughLines(binary_eagle, std_binary_lines, 1, CV_PI/180, 150, 0, 0);
    // drawHoughStd(color_binary_output, std_binary_lines, cv::Scalar(0,0,255), 1);

    // with canny ------------------------------------------------------------------------------------------------
    cv::Mat color_canny_output = color_eagle.clone();
    cv::Mat canny_dst;
    cv::Canny(binary_eagle, canny_dst, 50, 200, 3);
    
    std::vector<cv::Vec4i> plt_canny_lines;
    cv::HoughLinesP(canny_dst, plt_canny_lines, 1, CV_PI/180, 50, 50, 59);
    drawHoughPlt(color_canny_output, plt_canny_lines, cv::Scalar(0,255,0), 2);
    // // std Hough
    // std::vector<cv::Vec2f> std_canny_lines;
    // cv::HoughLines(canny_dst, std_canny_lines, 1, CV_PI/180, 150, 0, 0);
    // drawHoughStd(color_canny_output, std_canny_lines, cv::Scalar(0,0,255),1);

    // with Zhang Suen thinning: ---------------------------------------------------------------------------------
    cv::Mat color_thinning_output = cv::Mat().zeros(cv::Size(color_eagle.cols, color_eagle.rows), CV_8UC3); //color_eagle.clone();
    cv::Mat thinning_dst;
    cv::ximgproc::thinning(binary_eagle, thinning_dst);
    
    std::vector<cv::Vec4i> plt_thinning_lines;
    // cv::HoughLinesP(thinning_dst, plt_thinning_lines, 1, CV_PI/180, 50, 10, 10);
    cv::HoughLinesP(thinning_dst, plt_thinning_lines, 1, CV_PI/180, 10, 30, 5);
    drawHoughPlt(color_thinning_output, plt_thinning_lines, cv::Scalar(0,255,0), 2);
    // // std Hough
    // std::vector<cv::Vec2f> std_thinning_lines;
    // cv::HoughLines(thinning_dst, std_thinning_lines, 1, CV_PI/180, 150, 0, 0);
    // drawHoughStd(color_thinning_output, std_thinning_lines, cv::Scalar(0,0,255), 1);
    
    int count = plt_thinning_lines.size();
    std::vector<cv::Vec4i> out_lines;
    std::vector<int> mass;
    int tolerance = 50;
    for (size_t i = 0; i < count; i++)
    {
        bool similar = false;
        int count_out = out_lines.size();

        cv::Vec4i aux = plt_thinning_lines.at(i);

        for (size_t j = 0; (j < count_out) && (!similar); j++)
        {
            //Is it similar?
            cv::Vec4i center = out_lines.at(j);
            int mass_center = mass.at(j);
            if(aux[0] < (center[0] + tolerance) && aux[0] > (center[0] - tolerance) && 
            aux[1] < (center[1] + tolerance) && aux[1] > (center[1] - tolerance) &&
            aux[2] < (center[2] + tolerance) && aux[2] > (center[2] - tolerance) &&
            aux[3] < (center[3] + tolerance) && aux[3] > (center[3] - tolerance)
            ){
                //out_lines.at(j) = (center*mass_center + aux)/(mass_center + 1);

                cv::Vec4i adefesio = cv::Vec4i(min(aux[0],center[0]),min(aux[1],center[1]),max(aux[2],center[2]),max(aux[3],center[3]));
                out_lines.at(j) = adefesio;
                mass.at(j) = mass_center + 1;
                similar = true;
            }
            std::cout<<"Vectores: "<<center<<" y "<<aux<<" son similares "<<similar<<"\n";
        }

        if(!similar){
            out_lines.push_back(aux);
            mass.push_back(1);
        }
    }
    
    std::cout<<"OUTPUT LINES: ";
    for(size_t i=0; i<out_lines.size(); i++){
        cv::Vec4i l = out_lines[i];
        std::cout<<l<<"\n";
    }


    cv::Mat cluster_output = cv::Mat().zeros(cv::Size(color_eagle.cols, color_eagle.rows), CV_8UC3); //color_eagle.clone();
    
    drawHoughPlt(cluster_output, out_lines, cv::Scalar(0,255,255), 2);

    /*
        dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
        lines: A vector that will store the parameters (xstart,ystart,xend,yend) of the detected lines
        rho : The resolution of the parameter r in pixels. We use 1 pixel.
        theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
        threshold: The minimum number of intersections to "*detect*" a line
        minLineLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
        maxLineGap: The maximum gap between two points to be considered in the same line.
    */



    //cv::imshow("Input for Canny and thinning", binary_eagle);
    cv::imshow("Output thinning", thinning_dst);
    //cv::imshow("Input for Canny and thinning with closing", binary_closed);
    // cv::imshow("Algo input, canny", algo_input);
    // cv::imshow("Algo thinned input", algo_input_thinned);
    //cv::imshow("Binary Std Hough (red) plt (green)", color_binary_output);
    //cv::imshow("Canny Std Hough (red) plt (green)", color_canny_output);
    cv::imshow("Thinned Std Hough (red) plt (green)", color_thinning_output);
    cv::imshow("Clusterized output", cluster_output);

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

void drawHoughStd(cv::Mat canvas, std::vector<cv::Vec2f> std_lines, cv::Scalar color, int thickness){
    for(size_t i=0; i<std_lines.size(); i++){
        float rho = std_lines[i][0], theta = std_lines[i][1];
        cv::Point pt1, pt2;
        double a=cos(theta), b=sin(theta);
        double x0=a*rho, y0=b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        cv::line(canvas, pt1, pt2, color, thickness, cv::LINE_AA);
    }
    return;
}

void drawHoughPlt(cv::Mat canvas, std::vector<cv::Vec4i> plt_lines, cv::Scalar color, int thickness){
    for(size_t i=0; i<plt_lines.size(); i++){
        cv::Vec4i l = plt_lines[i];
        cv::Scalar p = cv::Scalar(color[0] + i*20 ,color[1] - i*20,color[2]);
        cv::line(canvas, cv::Point(l[0],l[1]), cv::Point(l[2],l[3]), p, thickness, cv::LINE_AA);
        //std::cout<<l<<"\n";
    }
}

int min(int num1, int num2){
    if(num1 < num2){
        return num1;
    }
    return num2;
}

int max(int num1, int num2){
    if(num1 > num2){
        return num1;
    }
    return num2;
}