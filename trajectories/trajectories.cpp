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

using namespace cv;
using namespace std;

std::vector<cv::Point> purePursuit(std::vector<cv::Point> left, std::vector<cv::Point> center, std::vector<cv::Point> right);
void mockups(int setNum, bool show_image);
std::vector<std::vector<cv::Point>> get_mockup(int mockup_id);
void sharpLeft(bool show_image);
void drawLane(cv::Mat img, std::vector<cv::Point> Lane, cv::Scalar color, int thickness);
void drawLaneFromCarCS(cv::Mat img, std::vector<cv::Point> Lane, cv::Scalar color, int thickness);
cv::Mat drawTrajectories();
cv::Mat baseImage();
cv::Point img2carCoordinate(cv::Point imgPoint);
cv::Point car2imgCoordinate(cv::Point carSpacePoint);

std::vector<cv::Point> img2carCoordinateVector(std::vector<cv::Point> line_in_img_coordinates);
std::vector<cv::Point> car2imgCoordinateVector(std::vector<cv::Point> line_in_car_coordinates);

int num_m_avg = 2; // min amount of points to average out the trajectory
int img_width = 480;
int img_height = 640;
int car_width = 200;
int car_height = 350;

int main(int argc, char** argv ) {
	std::vector<std::vector<cv::Point>> mockupLines;
	cv::Mat img;
	for (int i=0; i<8; i++){
		mockupLines = get_mockup(i);
		img = drawTrajectories(mockupLines);
		cv::imshow("Mockup", img);
		waitKey(0);
	}
    return 0;
}

cv::Mat drawTrajectories(std::vector<std::vector<cv::Point>> input){
	Mat image = baseImage();
	std::vector rightLine = input[2];
	std::vector centerLine = input[1];
	std::vector leftLine = input[0];

	drawLane(image, rightLine, Scalar(0,255,0), 2);
	drawLane(image, centerLine, Scalar(0,180,180), 1);
	drawLane(image, leftLine, Scalar(0,0,255), 2);

	std::vector<cv::Point> trajectory_car_cs = purePursuit(leftLine, centerLine, rightLine);	     // trajectory in car coordinates
	std::vector<cv::Point> trajectory_img_cs = car2imgCoordinateVector(trajectory_car_cs);				 // trajectory in image coordinates
	circle(image, trajectory_img_cs[trajectory_img_cs.size()-1], 7, Scalar(70, 180, 0), FILLED, LINE_8); // end-point
	line(image, trajectory_img_cs[trajectory_img_cs.size()-1], trajectory_img_cs[0],Scalar(70, 70, 0), 2); // line from current position to end point
	int numPoints = trajectory_car_cs.size();

	
	// // // calculated in car space
	// // OD/2 = [([last_x^2+last_y^2])^1/2]/2
	float Od2 = sqrt(pow(trajectory_car_cs[numPoints-1].x, 2) + pow(trajectory_car_cs[numPoints-1].y, 2))/2; // hlaf the distance from endpoint
	// // sin(alpha) = (last_y/2)/(OD/2)
	float sin_alpha = (trajectory_car_cs[trajectory_car_cs.size()-1].y/2)/Od2;
	if(sin_alpha!= 0){
		float R = Od2/sin_alpha;
		// cout << "R in mockup " << R << endl;
		cv::Point ICC = Point(0, R);
		cv::Point ICC_img_cs = car2imgCoordinate(ICC);
		circle(image, ICC_img_cs, abs(R), Scalar(150, 150, 0), 4, LINE_8);
	}else{
		line(image, trajectory_img_cs[0], trajectory_img_cs[numPoints-1],Scalar(150,150, 0), 2);
	}

	// // show the trajectory calculated
	for(int i=0; i<numPoints; i++){
		circle(image, trajectory_img_cs[i], 3, Scalar(255,0,255), -1, LINE_8);
	}
}

std::vector<cv::Point> purePursuit(std::vector<cv::Point> left, std::vector<cv::Point> center, std::vector<cv::Point> right){
	std::vector<cv::Point> trajectory;
	std::vector<cv::Point> right_car = img2carCoordinateVector(right);
	std::vector<cv::Point> center_car = img2carCoordinateVector(center);
	std::vector<cv::Point> left_car = img2carCoordinateVector(left);
	cv::Point lane_vector; // Vector from the furthest point in the centerline towards the right lane

	// estimate final point
	cv:Point end_point_car;
	float lane_width = 0;
	if (!right.empty() && !center.empty()){
		end_point_car.y = (center_car.at(0).y + right_car.at(0).y)/2;
		end_point_car.x = (center_car.at(0).x + right_car.at(0).x)/2;
	} else if (!left.empty() && !center.empty()){
		lane_vector = cv::Point((center_car.at(0).x-left_car.at(0).x)/2, (center_car.at(0).y-left_car.at(0).y)/2);
		end_point_car.y = center_car.at(0).y + lane_vector.y;
		end_point_car.x = center_car.at(0).x + lane_vector.x;
		cout << "Left: " << left_car.at(0) << " center: " << center_car.at(0) << " lane vector: " << lane_vector << endl;
	}else{
		end_point_car.x = img_height-10;
		end_point_car.y = 0;
	}


	// Calcular el radio: -------------------------------------------------------------------------------
	float Od2 = sqrt(pow(end_point_car.x, 2) + pow(end_point_car.y, 2))/2;
	float sin_alpha = (end_point_car.y/2)/Od2;
	float R;
	if(sin_alpha!= 0){
		R = Od2/sin_alpha;
		cout << "R in estimate " << R << endl;
		float R_2 = R/2;
		for (int x=0; x<end_point_car.x; x+=10){
			float y;
			y = sqrt(pow(R,2)-pow(x,2)) - abs(R);
			if (R>0){	// Right turn
				// y=-(R/2)+sqrt(pow((R/2),2)-pow(x,2));
				y = -y;
			}else{		// Left turn
				// y=-(R/2)-sqrt(pow((R/2),2)-pow(x,2));
				
			}
			trajectory.insert(trajectory.end(), Point(x,y));
			cout << "x: " << x << " y: " << y << endl;
		}
	}else{
		for (int x=0; x<end_point_car.x; x+=10){
			trajectory.insert(trajectory.end(), Point(x,0));
			// cout << "x: " << x << " y: " << 0 << endl;
		}
	}
	trajectory.insert(trajectory.end(), end_point_car);

	// cout << "lane width: " << lane_width << endl; // Achtung! Salen valores negativos :):
	
	return trajectory;
}

std::vector<std::vector<cv::Point>> get_mockup(int setNum){
	std::vector<cv::Point> leftLane;
	std::vector<cv::Point> centerLane;
	std::vector<cv::Point> rightLane;
	std::vector<std::vector<cv::Point>> output;

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
	
	output = {leftLane, centerLane, rightLane};
	return output;
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

	// // the cange in coordinates works
	// std::vector<cv::Point> car_CS_lane = img2carCoordinateVector(rightLane);
	// drawLane(image, car_CS_lane, Scalar(255,255,0), 1);
	// std::vector<cv::Point> img_CS_lane = car2imgCoordinateVector(car_CS_lane);
	// drawLane(image, img_CS_lane, Scalar(255,0,255), 2);

	cout << endl <<"Mockup #" << setNum << endl;
	std::vector<cv::Point> trajectory_car_cs = purePursuit(leftLane, centerLane, rightLane);	     // trajectory in car coordinates
	std::vector<cv::Point> trajectory_img_cs = car2imgCoordinateVector(trajectory_car_cs);				 // trajectory in image coordinates
	circle(image, trajectory_img_cs[trajectory_img_cs.size()-1], 7, Scalar(70, 180, 0), FILLED, LINE_8); // end-point
	line(image, trajectory_img_cs[trajectory_img_cs.size()-1], trajectory_img_cs[0],Scalar(70, 70, 0), 2); // line from current position to end point
	int numPoints = trajectory_car_cs.size();
	cout <<"Size trajectory " << numPoints << endl;
	cout <<"End point car coordinates: " << trajectory_car_cs[numPoints-1].x << " " << trajectory_car_cs[numPoints-1].y << endl;
	
	// // // calculated in car space
	// // OD/2 = [([last_x^2+last_y^2])^1/2]/2
	float Od2 = sqrt(pow(trajectory_car_cs[numPoints-1].x, 2) + pow(trajectory_car_cs[numPoints-1].y, 2))/2; // hlaf the distance from endpoint
	// // sin(alpha) = (last_y/2)/(OD/2)
	float sin_alpha = (trajectory_car_cs[trajectory_car_cs.size()-1].y/2)/Od2;
	if(sin_alpha!= 0){
		float R = Od2/sin_alpha;
		cout << "R in mockup " << R << endl;
		cv::Point ICC = Point(0, R);
		cv::Point ICC_img_cs = car2imgCoordinate(ICC);
		circle(image, ICC_img_cs, abs(R), Scalar(150, 150, 0), 4, LINE_8);
	}else{
		line(image, trajectory_img_cs[0], trajectory_img_cs[numPoints-1],Scalar(150,150, 0), 2);
	}

	// // show the trajectory calculated
	for(int i=0; i<numPoints; i++){
		circle(image, trajectory_img_cs[i], 3, Scalar(255,0,255), -1, LINE_8);
	}

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

	std::vector<cv::Point> trajectory = purePursuit(leftLane, centerLane, rightLane);
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

void drawLaneFromCarCS(cv::Mat img, std::vector<cv::Point> lane, cv::Scalar color, int thickness) {
	std::vector<cv::Point> lane_img_CS = car2imgCoordinateVector(lane);
	if (lane_img_CS.size() != 0){
		for (int i=1; i<lane_img_CS.size(); i++){
			line(img, lane_img_CS[i-1], lane_img_CS[i], color, thickness, LINE_8);
		}
	}
	return;
}

cv::Mat baseImage(){
	// lane_width = 300px | just for visualization
	Mat image(img_height, img_width, CV_8UC3, Scalar(0,0,0));
	cv::Point pt1(img_width/2-car_width/2, img_height-car_height), pt2(img_width/2+car_width/2, img_height);
	cv::rectangle(image, pt1, pt2, cv::Scalar(150, 150, 150));
	cv::arrowedLine(image, cv::Point(img_width/2,img_height), cv::Point(img_width/2, img_height-50), cv::Scalar(0,0,255), 4);
	cv::arrowedLine(image, cv::Point(img_width/2,img_height), cv::Point(img_width/2-50, img_height), cv::Scalar(0,255,0), 4);
	int thickness=1;
	return image;
}

cv::Point img2carCoordinate(cv::Point imgPoint){
	cv::Point point_in_car_coordinates;
	point_in_car_coordinates.x = img_height-imgPoint.y;
	point_in_car_coordinates.y = imgPoint.x - img_width/2;
	return point_in_car_coordinates;
}

cv::Point car2imgCoordinate(cv::Point carSpacePoint){
	cv::Point point_in_img_coordinates;
	point_in_img_coordinates.x = carSpacePoint.y + img_width/2;
	point_in_img_coordinates.y = img_height - carSpacePoint.x;
	return point_in_img_coordinates;
}

std::vector<cv::Point> img2carCoordinateVector(std::vector<cv::Point> line_in_img_coordinates){
	std::vector<cv::Point> vector_car_coordinates;
	if(!line_in_img_coordinates.empty()){
		for (int i=0; i<line_in_img_coordinates.size(); i++){
			cv::Point point_car = img2carCoordinate(line_in_img_coordinates.at(i));
			// point_car.x = img_height-line_in_img_coordinates.at(i).y;
			// point_car.y = line_in_img_coordinates.at(i).x - img_width/2;
			vector_car_coordinates.insert(vector_car_coordinates.end(), point_car);
		}
	}
	return vector_car_coordinates;
}

std::vector<cv::Point> car2imgCoordinateVector(std::vector<cv::Point> line_in_car_coordinates){
	std::vector<cv::Point> vector_img_coordinates;
	if(!line_in_car_coordinates.empty()){
		for (int i=0; i<line_in_car_coordinates.size(); i++){
			cv::Point point_img = car2imgCoordinate(line_in_car_coordinates.at(i));
			// point_img.x = line_in_car_coordinates.at(i).y + img_width/2;
			// point_img.y = img_height - line_in_car_coordinates.at(i).x;
			vector_img_coordinates.insert(vector_img_coordinates.end(), point_img);
		}
	}
	return vector_img_coordinates;
}
