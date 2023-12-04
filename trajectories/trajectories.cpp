// C++ program for the above approach 
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp> 
  
// Library to include for 
// drawing shapes 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

std::vector<cv::Point> estimateTrajectory(std::vector<cv::Point> left, std::vector<cv::Point> center, std::vector<cv::Point> right);
void mockups(int setNum, bool show_image);
void sharpLeft(bool show_image);
void drawLane(cv::Mat img, std::vector<cv::Point> Lane, cv::Scalar color, int thickness);
cv::Mat baseImage();

int main(int argc, char** argv ) {
	sharpLeft(true);
	for (int i=0; i<8; i++){
		mockups(i, true);
	}
    return 0;
}


std::vector<cv::Point> estimateTrajectory(std::vector<cv::Point> left, std::vector<cv::Point> center, std::vector<cv::Point> right){
	std::vector<cv::Point> trajectory;
	cv::Point pf;		// for the easiest trayectory planning
	if (!right.empty() && !center.empty()){
		pf.x = (right[0].x + center[0].x)/2;
		pf.y = 10;
		trajectory.push_back(pf);
	}else{
		pf.x = 0;
		pf.y = 0;
		trajectory.push_back(pf);
	}
	return trajectory;
}


void mockups(int setNum, bool show_image){
	std::vector<cv::Point> leftLane;
	std::vector<cv::Point> centerLane;
	std::vector<cv::Point> rightLane;

	switch (setNum){
		case 0:
			leftLane = {Point(185,0), Point(203,130), Point(224,252), Point(217,391)};
			centerLane = {Point(356,0), Point(394,130), Point(426,290), Point(441,477)};
			rightLane = {};
			break;
		case 1:
			leftLane = {Point(0,362), Point(141,399), Point(370,414), Point(463,396)};
			centerLane = {Point(75,134), Point(165,161), Point(277,172), Point(378,167)};
			rightLane = {};
			break;
		case 2:
			leftLane = {};
			centerLane = {Point(46,81), Point(96,188), Point(121,346), Point(120,466)};
			rightLane = {Point(248,4), Point(310,138), Point(347,266), Point(361,389), Point(367,577)};
			break;
		case 3:
			leftLane = {Point(4,163), Point(90,272), Point(154,339), Point(377,502), Point(467,544)};
			centerLane = {Point(171,9), Point(245,106), Point(310,172), Point(402,237), Point(471,269)};
			rightLane = {Point(431,-1), Point(470,31)};
			break;
		case 4:
			leftLane = {Point(181,5), Point(91,79), Point(-1,117)};
			centerLane = {Point(383,3), Point(374,40), Point(338,19), Point(295,168), Point(226,227), Point(159,270), Point(67,312), Point(-3,329)};
			rightLane = {Point(470,261), Point(405,334), Point(320,420), Point(163,509), Point(9,560)};
			break;
		case 5:
			leftLane = {Point(33,19), Point(4,131)};
			centerLane = {Point(221,89), Point(202,168), Point(166,242), Point(133,323), Point(91,421), Point(45,517)};
			rightLane = {Point(435,6), Point(407,191), Point(379,314), Point(340,453)};
			break;
		case 6:
			leftLane = {Point(9,197), Point(103,301), Point(161,398), Point(211,630)};
			centerLane = {Point(163,42), Point(222,106), Point(290,174), Point(345,248), Point(395,344), Point(431,442), Point(450,558), Point(458,631)};
			rightLane = {Point(412,1), Point(468,62)};
			break;
		default:
			leftLane = {};
			centerLane = {};
			rightLane = {};
			break;
	}

	Mat image = baseImage();
	drawLane(image, rightLane, Scalar(0,255,0), 2);
	drawLane(image, centerLane, Scalar(0,180,180), 1);
	drawLane(image, leftLane, Scalar(0,0,255), 2);

	std::vector<cv::Point> trajectory = estimateTrajectory(leftLane, centerLane, rightLane);
	circle(image, trajectory[0], 7, Scalar(70, 180, 0), FILLED, LINE_8);
	
	string file_name = "mockup_";
	file_name.append(std::to_string(setNum));
	file_name.append(".png");
	imwrite(file_name , image);
	if (show_image) {
		cv::imshow(file_name, image);
		waitKey(0);
	}
	return;
}

void sharpLeft(bool show_image){
	std::vector<cv::Point> leftLane = {Point(0,140), Point(153,210), Point(200,337), Point(200,525)};
	std::vector<cv::Point> centerLane = {Point(196,0), Point(342,149), Point(393,337), Point(393,525)};
	std::vector<cv::Point> rightLane = {Point(484,0), Point(570,167), Point(594,337), Point(594,525)};
	
	Mat image = baseImage();
	drawLane(image, rightLane, Scalar(0,255,0), 2);
	drawLane(image, centerLane, Scalar(0,180,180), 1);
	drawLane(image, leftLane, Scalar(0,0,255), 2);

	std::vector<cv::Point> trajectory = estimateTrajectory(leftLane, centerLane, rightLane);
	circle(image, trajectory[0], 7, Scalar(70, 180, 0), FILLED, LINE_8);
	
	imwrite("sharpLeft.png", image);
	if (show_image) {
		cv::imshow("sharp left", image);
		waitKey(0);
	}
	return;
}

void drawLane(cv::Mat img, std::vector<cv::Point> lane, cv::Scalar color, int thickness) {
	if (lane.size() != 0){
		for (int i=1; i<lane.size(); i++){
			line(img, lane[i-1], lane[i], color, thickness, LINE_8);
		}
	}
	return;
}

cv::Mat baseImage(){
	// lane_width = 300px | just for visualization
	int width = 480;
	int height = 640;
	int car_width = 200;
	int car_height = 350;
	Mat image(height, width, CV_8UC3, Scalar(0,0,0));
	cv::Point pt1(width/2-car_width/2, height-car_height), pt2(width/2+car_width/2, height);
	cv::rectangle(image, pt1, pt2, cv::Scalar(150, 150, 150));
	cv::arrowedLine(image, cv::Point(width/2,height), cv::Point(width/2, height-50), cv::Scalar(0,0,255), 4);
	cv::arrowedLine(image, cv::Point(width/2,height), cv::Point(width/2-50, height), cv::Scalar(0,255,0), 4);
	int thickness=1;
	return image;
}
