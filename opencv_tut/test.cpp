// C++ program for the above approach 
#include <iostream>
#include <fstream>
#include <stdio.h>

#include "utils.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv ) {
	cout << "line 1 from test file" << endl;
	// sharpLeft(true);
	Mat image_test(150, 150, CV_8UC3, Scalar(0,255,0));
	cv::imshow("img test", image_test);
	waitKey(0);
    return 0;
}
