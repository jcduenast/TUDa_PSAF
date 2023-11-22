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

std::vector<cv::Point> estimateTrayectory(std::vector<cv::Point> left, std::vector<cv::Point> center, std::vector<cv::Point> right);
void straight(bool show_image);
void sharpLeft(bool show_image);
void slightRight(bool show_image);
void drawLane(cv::Mat img, std::vector<cv::Point> Lane, cv::Scalar color, int thickness);
cv::Mat baseImage();

int main(int argc, char** argv ) {
	straight(true);
	sharpLeft(true);
	slightRight(true);
    return 0;
}

std::vector<cv::Point> estimateTrayectory(std::vector<cv::Point> left, std::vector<cv::Point> center, std::vector<cv::Point> right){
	std::vector<cv::Point> trayectory;
	/*
	estimate width of the lane
	if none available... pick a line to follow
	*/
	cv::Point pf;		// for the easiest trayectory planning


	pf.x = (right[0].x + center[0].x)/2;
	pf.y = 10;
	trayectory.push_back(pf);
	return trayectory;
}

void straight(bool show_image){
	std::vector<cv::Point> leftLane = {Point(100, 0), Point(89, 100), Point(100, 200), Point(100, 300)};
	std::vector<cv::Point> centerLane = {Point(160, 0), Point(160, 100), Point(160, 200), Point(160, 300)};
	std::vector<cv::Point> rightLane = {Point(460, 0), Point(460, 100), Point(460, 200), Point(460, 300)};
	
	Mat image = baseImage();
	drawLane(image, rightLane, Scalar(0,255,0), 2);
	drawLane(image, centerLane, Scalar(0,180,180), 1);
	drawLane(image, leftLane, Scalar(0,0,255), 2);

	std::vector<cv::Point> trayectory = estimateTrayectory(leftLane, centerLane, rightLane);
	circle(image, trayectory[0], 7, Scalar(70, 180, 0), FILLED, LINE_8);
	
	imwrite("straight.png", image);
	if (show_image) {
		cv::imshow("Straight", image);
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

	std::vector<cv::Point> trayectory = estimateTrayectory(leftLane, centerLane, rightLane);
	circle(image, trayectory[0], 7, Scalar(70, 180, 0), FILLED, LINE_8);
	
	imwrite("sharpLeft.png", image);
	if (show_image) {
		cv::imshow("sharp left", image);
		waitKey(0);
	}
	return;
}

void slightRight(bool show_image){
	std::vector<cv::Point> leftLane = {};
	std::vector<cv::Point> centerLane = {Point(269,0), Point(187,179), Point(169,338), Point(187,525)};
	std::vector<cv::Point> rightLane = {Point(678,6), Point(543,179), Point(505,337), Point(543,525)};
	
	Mat image = baseImage();
	drawLane(image, rightLane, Scalar(0,255,0), 2);
	drawLane(image, centerLane, Scalar(0,180,180), 1);
	drawLane(image, leftLane, Scalar(0,0,255), 2);

	std::vector<cv::Point> trayectory = estimateTrayectory(leftLane, centerLane, rightLane);
	circle(image, trayectory[0], 7, Scalar(70, 180, 0), FILLED, LINE_8);
	
	imwrite("slightRight.png", image);
	if (show_image) {
		cv::imshow("slight right", image);
		waitKey(0);
	}
	return;
}

void drawLane(cv::Mat img, std::vector<cv::Point> lane, cv::Scalar color, int thickness) {
	if (lane.size() != 0){
		for (int i=1; i<lane.size(); i++){	// empty Vector not considered yet !!!!!
			line(img, lane[i-1], lane[i], color, thickness, LINE_8);
		}
	}
	return;
}

cv::Mat baseImage(){
	// lane_width = 300px | just for visualization
	int width = 700;
	int height = 700;
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
