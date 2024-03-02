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

// para el manejo de strings
#include <sstream>
#include <iomanip>

#include <opencv2/ximgproc.hpp>

// for blobs
#include <opencv2/features2d.hpp>

#include "main.hpp"

cv::Mat perspective_forward();
cv::Mat perspective_back(cv::Mat img);
cv::Mat process_eagle_img(cv::Mat eagle_img);
cv::Mat get_eagle_view(cv::Mat img_in);
cv::Mat process_raw_classified(cv::Mat raw, cv::Mat classiffied);

int main (int argc, char *argv[]){
    // cv::Mat projected = perspective_forward();
    // cv::imshow("Forward", projected);
    // cv::imshow("Back", perspective_back(projected));
    // cv::waitKey(0);

    int set = 10;
    if(argc > 1){
        set = atoi(argv[1]);
    }

    int frame = 0;
    std::string root_path, cam_img_name, eagle_img_name;
    cv::Mat og_img, parking_img, eagle_classified;

    std::string local_root_path = "/home/ubi/usb/";

    std::stringstream ss;
    ss << std::setw(2) << std::setfill('0') << set;
    std::string run_id_string = ss.str();
    root_path = local_root_path + "invalido" + run_id_string + "/";

    for(;; frame++){
        cam_img_name = root_path + "raw_img_" + std::to_string(frame) + ".jpg";
        eagle_img_name = root_path + "classification_" + std::to_string(frame) + ".jpg";
        std::cout << "Frame: " << std::to_string(frame) << " at: " << cam_img_name << std::endl;
        og_img = cv::imread(cam_img_name);
        eagle_classified = cv::imread(eagle_img_name);
        
        cv::imshow("Inverse wrap", process_eagle_img(eagle_classified));
        cv::waitKey(0);
    }

    return 0;
}

cv::Mat process_raw_classified(cv::Mat raw, cv::Mat classiffied){
    cv::Mat res;

    return res;
}

cv::Mat process_eagle_img(cv::Mat eagle_img){
    cv::Mat res;
    cv::Mat homography;
    homography = (cv::Mat(3, 3, CV_64F, homography_data_juan_daniel_hd)).clone();
    cv::warpPerspective(eagle_img, res, homography, cv::Size(640, 490), cv::WARP_INVERSE_MAP);
    return res;    
}

cv::Mat get_eagle_view(cv::Mat img_in){
    cv::Mat projected;
    cv::Mat homography;
    homography = (cv::Mat(3, 3, CV_64F, homography_data_juan_daniel_hd)).clone();
    cv::warpPerspective(img_in, projected, homography, cv::Size(640, 490));
    return projected;
}

cv::Mat perspective_forward(){
    cv::Mat res;

    // cv::Mat img = cv::imread("data/raw_img_347.jpg");
    // cv::Point2f srcPoints[] = {
    //     cv::Point(471,77),
    //     cv::Point(513, 75),
    //     cv::Point(470, 111),
    //     cv::Point(508, 114)
    // };
    // cv::Point2f dstPoints[] = {
    //     cv::Point(0,0),
    //     cv::Point(150,0),
    //     cv::Point(0, 150),
    //     cv::Point(150,150)
    // };

    cv::Mat img = cv::imread("data/test_img.jpg");
    cv::Point2f srcPoints[] = {
        cv::Point(209,145),
        cv::Point(384,127),
        cv::Point(222, 275),
        cv::Point(432, 242)
    };
    cv::Point2f dstPoints[] = {
        cv::Point(0,0),
        cv::Point(150,0),
        cv::Point(0, 150),
        cv::Point(150,150)
    };

    cv::Mat M = cv::getPerspectiveTransform(srcPoints, dstPoints);
    cv::warpPerspective(img, res, M, cv::Size(640,360), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    cv::circle(res, cv::Point(75,75), 50, cv::Scalar(255,0,0), 15);
    // std::cout << M;

    return res;
}

cv::Mat perspective_back(cv::Mat img){
    cv::Mat res;

    cv::Point2f srcPoints[] = {
        cv::Point(209,145),
        cv::Point(384,127),
        cv::Point(222, 275),
        cv::Point(432, 242)
    };
    cv::Point2f dstPoints[] = {
        cv::Point(0,0),
        cv::Point(150,0),
        cv::Point(0, 150),
        cv::Point(150,150)
    };

    cv::Mat M = cv::getPerspectiveTransform(srcPoints, dstPoints);
    cv::warpPerspective(img, res, M, cv::Size(640,360), cv::WARP_INVERSE_MAP);

    return res;
}