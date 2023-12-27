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


void draw_line_and_circle(cv::Mat img_in){
    cv::line(img_in,cv::Point(0,0), cv::Point(100,100), cv::Scalar(100,200,255), 3);
    cv::circle(img_in, cv::Point(0,0), 75, cv::Scalar(255,50,150), 3);
    return;
}