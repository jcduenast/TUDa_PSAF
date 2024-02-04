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

// to measure execution time
#include <chrono>

// using namespace cv;
// using namespace std;

cv::RNG rng(12345);

cv::Mat get_eagle_view(cv::Mat img_in, int mode);
cv::Mat inf_processing(cv::Mat camera_raw_color, int mode);
cv::Mat car_on_projection(cv::Mat img_in, int mode);
cv::Mat final_on_og(cv::Mat img_final, cv::Mat img_og); // img_final U8C1
std::vector<std::vector<cv::Point>> proc_proposal(cv::Mat camera_raw_color);
void lineClasification(cv::Mat raw_color_camera);
std::vector<std::vector<std::vector<cv::Point>>> lineClasification_old(cv::Mat raw_color_camera);
void test_algo(int mode, int set);
std::vector<bool> filterConnectedCenterLines(std::vector<std::vector<cv::Point>> candidatesContours,std::vector<bool> isCandidate);
bool isAligned(std::vector<cv::Point> area1, std::vector<cv::Point> area2);
float getPositionAtBottom(std::vector<cv::Point> line); 
std::vector<bool> filterConnectedCenterLines(std::vector<std::vector<cv::Point>> candidatesContours,std::vector<bool> isCandidate);
bool isAligned(std::vector<cv::Point> area1, std::vector<cv::Point> area2);
float getPositionAtBottom(std::vector<cv::Point> line); 
cv::Point getMaxYPoint(std::vector<cv::Point> region);
std::vector<cv::Point> new_trajectory(std::vector<std::vector<std::vector<cv::Point>>> lines);

int main (){
    test_algo(3,1);
    return 0;
}

void lineClasification(cv::Mat raw_color_camera){

    const int MIN_WIDTH_CENTER_LINE = 7;
    const int MAX_WIDTH_CENTER_LINE = 25;
    const int MIN_LENGTH_CENTER_LINE = 50;
    const int MAX_LENGTH_CENTER_LINE = 100;
    const int MIN_AREA_CENTER_LINE = 100;
    const int MAX_AREA_CENTER_LINE = 400;
    const int MIN_AREA_LATERAL_LINE = 500;
    const int WIDTH_IMAGE = 640;

    cv::Mat gray, blurred, binary, binary_eagle;
    cv::cvtColor(raw_color_camera, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 0, 0, cv::BORDER_DEFAULT);
    int block_size=101, const_subtrahend=-50;
    cv::adaptiveThreshold(blurred, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block_size, const_subtrahend);
    int mode=3; // 2 para la transformación más amplia
    binary_eagle = get_eagle_view(binary, mode);

    std::vector<std::vector<cv::Point>> leftLineRegion;
    std::vector<std::vector<cv::Point>> centerLinesRegion;
    std::vector<std::vector<cv::Point>> rightLineRegion;
    int leftLineIndex;
    int rightLineIndex;

    // Contour detection --------------------------------------------------------------
    std::vector<std::vector<cv::Point>> cnt;        // Here will be the ones bigger than 100px
    std::vector<std::vector<cv::Point>> cntAll;     // Here all the contours will be stored for the first clasification
    std::vector<cv::Vec4i> hierarchyAll;
    std::vector<cv::Vec4i> hierarchy;
    // cv::findContours(binary_eagle, cntAll, hierarchyAll, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE); // usefull for finding the parking spot
    cv::findContours(binary_eagle, cntAll, hierarchyAll, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE); // gets only the external contours

    // filter small contours -------------------------------------------------------------
    for (int i=0; i<cntAll.size(); i++)     if (cntAll[i].size() > 100) cnt.insert(cnt.begin(), cntAll[i]);    

    // Center lines just as center lines on their own -------------------------------------------------------
    
    cv::Mat centerLinesSegmentation = cv::Mat().zeros(binary_eagle.size(), CV_8UC3);    // to show the result of the segmentation
    std::vector<std::vector<cv::Point>> cntCenterCandidates;

    // Drawing all contours in gray, the center ones will be later resalted
    for (int i=0; i<cntAll.size(); i++) cv::drawContours(centerLinesSegmentation, cntAll, i, cv::Scalar(70,70,70), cv::FILLED, cv::LINE_8);

    for (int i=0; i<cntAll.size(); i++){    // sorting out by size of contour
        if (cntAll[i].size() > MIN_AREA_CENTER_LINE && cntAll[i].size() < MAX_AREA_CENTER_LINE){
            cntCenterCandidates.insert(cntCenterCandidates.begin(), cntAll[i]);
        }
    }

    std::vector<bool> boolCenter(cntCenterCandidates.size());   // if index set to true, will be considered as a center line as the algorithm progresses

    std::vector<cv::Rect> centerBoundRect(cntCenterCandidates.size());
    std::vector<cv::RotatedRect> centerRotBoundRect(cntCenterCandidates.size());
    cv::Point2f centerRotRectPoints_aux[4];
    int center_minDst, center_maxDst;

    //Label, draw and filter by size the center lines
    for (int i=0; i<cntCenterCandidates.size(); i++){
        centerBoundRect[i] = cv::boundingRect(cntCenterCandidates[i]);
        cv::minAreaRect(cntCenterCandidates[i]).points(centerRotRectPoints_aux);
        center_minDst = std::min(cv::norm(centerRotRectPoints_aux[0]-centerRotRectPoints_aux[1]), cv::norm(centerRotRectPoints_aux[1]-centerRotRectPoints_aux[2]));
        center_maxDst = std::max(cv::norm(centerRotRectPoints_aux[0]-centerRotRectPoints_aux[1]), cv::norm(centerRotRectPoints_aux[1]-centerRotRectPoints_aux[2]));
        //Labeling the size
        cv::putText(centerLinesSegmentation, std::to_string(center_minDst) + " " + std::to_string(center_maxDst)+ " i:" + std::to_string(i),
                    cv::Point(centerBoundRect[i].tl().x-15, centerBoundRect[i].br().y+15), 
                    cv::FONT_HERSHEY_COMPLEX_SMALL , 0.7, CV_RGB(255,255,255), 1, cv::LINE_8, false);
        //Drawing the minimum bounding box
        for (int j=0; j<4; j++) {
            cv::line(centerLinesSegmentation, centerRotRectPoints_aux[j], centerRotRectPoints_aux[(j+1)%4], cv::Scalar(150,150,0));
        }
        //Drawing the rect bounding box
        cv::rectangle(centerLinesSegmentation, centerBoundRect[i].tl(), centerBoundRect[i].br(), cv::Scalar(255,255,255), 1, cv::LINE_8);
        //Verifying the width and length for each center line
        if(center_minDst > MIN_WIDTH_CENTER_LINE && center_minDst < MAX_WIDTH_CENTER_LINE 
            && center_maxDst > MIN_LENGTH_CENTER_LINE && center_maxDst < MAX_LENGTH_CENTER_LINE){
            boolCenter[i] = true;
        }else{
            boolCenter[i] = false;
        }
    }

    //Check for each center line if its line vector connects it to the anothers lines

    boolCenter = filterConnectedCenterLines(cntCenterCandidates,boolCenter);

    //Create a region for all the connected center lines
    std::vector<cv::Point> centerRegion;
    for (size_t i = 0; i < cntCenterCandidates.size(); i++)
    {
        if(boolCenter[i] == true){
            std::vector<cv::Point> auxRegion = cntCenterCandidates.at(i);
            for (size_t j = 0; j < auxRegion.size(); j++)
            {
                centerRegion.push_back(auxRegion.at(j));
            }
        }
    }

    // Revisando la distribución estadística horizontal (en eje Y) No funciona, [abortado]

    //Check the biggest contours and defining left and right line
    std::vector<int> indexLongLines;
    bool isLeftLine = false;
    bool isRightLine = false;
    //Selecting the biggest than the minimum area defined
     for (int i=0; i<cntAll.size(); i++){
        if (cntAll[i].size() > MIN_AREA_LATERAL_LINE){
            indexLongLines.push_back(i);
        }
    }
    //Check which one is placed nearest to the center lines and its relative position

    //Check if there are detected center lines
    if(centerRegion.size()>2){
        float centerLinePosition = getPositionAtBottom(centerRegion);
        float leftLinePosition = -999999999999;
        float rightLinePosition = 99999999999;
        for (size_t i = 0; i < indexLongLines.size(); i++)
        {
            float aux = getPositionAtBottom(cntAll[indexLongLines.at(i)]); 
            //std::cout<<"POS LINE:\t"<<aux<<"\tPOS CENTER:\t"<<centerLinePosition<<"\n";
            if(aux > leftLinePosition && aux < centerLinePosition){
                leftLineIndex = indexLongLines.at(i);
                leftLinePosition = aux;
                isLeftLine = true;
            }
            else if (aux < rightLinePosition && aux > centerLinePosition)
            {
                rightLineIndex = indexLongLines.at(i);
                rightLinePosition = aux;
                isRightLine = true;
            }        
        }
    }

    // Dibujando ya los que sí quedaron finalmente-------------------------------------------------------
    for (int i=0; i<cntCenterCandidates.size(); i++){
        if (boolCenter[i]){
            cv::drawContours(centerLinesSegmentation, cntCenterCandidates, i, cv::Scalar(255,255,0), cv::FILLED, cv::LINE_8);
        }
    }
    if(isRightLine){
        cv::drawContours(centerLinesSegmentation, cntAll, rightLineIndex, cv::Scalar(0,0,255), cv::FILLED, cv::LINE_8);
        cv::Point aux = getMaxYPoint(cntAll[rightLineIndex]);
        std::cout<<"\nEl borde inferior del derecho es: "<<aux.x<<","<<aux.y<<"\n";
    }
    if(isLeftLine){
        cv::drawContours(centerLinesSegmentation, cntAll, leftLineIndex, cv::Scalar(0,255,0), cv::FILLED, cv::LINE_8);
        cv::Point aux = getMaxYPoint(cntAll[leftLineIndex]);
        std::cout<<"\nEl borde inferior del izquierdo es: "<<aux.x<<","<<aux.y<<"\n";
    }

    // for parking info processing
    cv::imshow("Center lines segmentation", centerLinesSegmentation);
    cv::waitKey(0);
    // std::cout << std::endl;
}

void test_algo(int mode, int set){
    int frame = 0;
    std::string root_path, cam_img_name, prc_img_name;
    cv::Mat own_processed, car_processed, og_img, eagle_view_color, own_processed_overlay;

    std::stringstream ss;
    ss << std::setw(2) << std::setfill('0') << set;
    std::string run_id_string = ss.str();

    // Ahora estamos con herr lisiado 1 y 2

    std::string local_root_path = "/home/ubi/usb/";  // pa' camilo
    // std::string local_root_path = "/home/daniel/Documentos/TU/PSAF/TUDa_PSAF/camera_processing/";  // pa' Daniel

    root_path = local_root_path + "lisiado" + std::to_string(set) + "/";

    // root_path = "/home/ubi/usb/run" + run_id_string + "/";
    // root_path = "/home/ubi/TUDa_PSAF/camera_processing/test/"; // path for camilo
    // root_path = "/home/daniel/Documentos/TU/PSAF/TUDa_PSAF/camera_processing/test/"; // path for Daniel

    bool record_old = false;

    for(;; frame++){
        cam_img_name = root_path + "raw_img_" + std::to_string(frame) + ".jpg";        // hasta la 2 está con png, de ahí en adelante con .jpg
        std::cout << "Frame: " << std::to_string(frame) << " at: " << cam_img_name << std::endl;
        og_img = cv::imread(cam_img_name);                                              // cargar la imagen de la camara a color
        lineClasification(og_img);
        

        if (record_old){
            std::vector<std::vector<std::vector<cv::Point>>> output = lineClasification_old(og_img);
            cv::Mat rightLines = cv::Mat().zeros(cv::Size(640,640), CV_8UC3);     // where the result of the algorithm will be visualized
            for (int i=0; i<output[0].size(); i++) cv::drawContours(rightLines, output[0], i, cv::Scalar(255,255,255), cv::FILLED, cv::LINE_8);
            cv::imwrite(root_path + "lines_right_" + std::to_string(frame) + ".jpg", rightLines);
            
            cv::Mat centerLines = cv::Mat().zeros(cv::Size(640,640), CV_8UC3);     // where the result of the algorithm will be visualized
            for (int i=0; i<output[1].size(); i++) cv::drawContours(centerLines, output[1], i, cv::Scalar(255,255,255), cv::FILLED, cv::LINE_8);
            cv::imwrite(root_path + "lines_center_" + std::to_string(frame) + ".jpg", centerLines);
            
            cv::Mat leftLines = cv::Mat().zeros(cv::Size(640,640), CV_8UC3);     // where the result of the algorithm will be visualized
            for (int i=0; i<output[2].size(); i++) cv::drawContours(leftLines, output[2], i, cv::Scalar(255,255,255), cv::FILLED, cv::LINE_8);
            cv::imwrite(root_path + "lines_left_" + std::to_string(frame) + ".jpg", leftLines);
        }
    }
    return;
}

std::vector<cv::Point> new_trajectory(std::vector<std::vector<std::vector<cv::Point>>> lines){
    std::vector<cv::Point> output;  // dos puntitos na' más

    return output;
}

cv::Mat final_on_og(cv::Mat img_final, cv::Mat img_og){
    cv::Mat output;
    cv::Mat final_not;
    cv::bitwise_not(img_final, final_not);
    img_og.copyTo(output, final_not);
    return output;
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
        case 3:     // la nueva, en HD y la cámara apuntando para abajo
            homography = (cv::Mat(3, 3, CV_64F, homography_data_juan_daniel_hd)).clone();
            cv::warpPerspective(img_in, projected, homography, cv::Size(640, 490));
        break;
        default:
            homography = (cv::Mat(3, 3, CV_64F, homography_data_juan_daniel_hd)).clone();
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

std::vector<std::vector<std::vector<cv::Point>>> lineClasification_old(cv::Mat raw_color_camera){
    
    const int MIN_WIDTH_CENTER_LINE = 7;
    const int MAX_WIDTH_CENTER_LINE = 25;
    const int MIN_LENGTH_CENTER_LINE = 50;
    const int MAX_LENGTH_CENTER_LINE = 100;
    const int MIN_AREA_CENTER_LINE = 100;
    const int MAX_AREA_CENTER_LINE = 400;
    const int MIN_AREA_LATERAL_LINE = 500;
    const int WIDTH_IMAGE = 640;

    cv::Mat gray, blurred, binary, binary_eagle;
    cv::cvtColor(raw_color_camera, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 0, 0, cv::BORDER_DEFAULT);
    int block_size=101, const_subtrahend=-50;
    cv::adaptiveThreshold(blurred, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block_size, const_subtrahend);
    int mode=3; // 2 para la transformación más amplia
    binary_eagle = get_eagle_view(binary, mode);

    std::vector<std::vector<cv::Point>> leftLineRegion;
    std::vector<std::vector<cv::Point>> centerLinesRegion;
    std::vector<std::vector<cv::Point>> rightLineRegion;
    int leftLineIndex;
    int rightLineIndex;

    // Contour detection --------------------------------------------------------------
    std::vector<std::vector<cv::Point>> cnt;        // Here will be the ones bigger than 100px
    std::vector<std::vector<cv::Point>> cntAll;     // Here all the contours will be stored for the first clasification
    std::vector<cv::Vec4i> hierarchyAll;
    std::vector<cv::Vec4i> hierarchy;
    // cv::findContours(binary_eagle, cntAll, hierarchyAll, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE); // usefull for finding the parking spot
    cv::findContours(binary_eagle, cntAll, hierarchyAll, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE); // gets only the external contours

    cv::Mat result = cv::Mat().zeros(binary_eagle.size(), CV_8UC3);     // where the result of the algorithm will be visualized

    // Draw all contours in grey, will be overwritten when classified
    for (int i=0; i<cntAll.size(); i++) cv::drawContours(result, cntAll, i, cv::Scalar(100,100,100), cv::FILLED, cv::LINE_8);

    // filter small contours -------------------------------------------------------------
    for (int i=0; i<cntAll.size(); i++)     if (cntAll[i].size() > 100) cnt.insert(cnt.begin(), cntAll[i]);

    // finding the contour (cnt) with the largest area to tag them as either left or right lines --------------------------------------
    int indexLargestCnt = -1, index2ndLargestCnt = -1;
    int areaLargestCnt = 600, area2ndLargestCnt = 600;        // Side lines should be at least 600 big, normally arround +900

    for (int i=0; i<cnt.size(); i++){
        // std::cout << "checking cnt " << std::to_string(i) << " with " << std::to_string(cnt[i].size()) << " Largest area: " << std::to_string(areaLargestCnt);
        // std::cout <<" 2nd largest area: " << std::to_string(area2ndLargestCnt) << std::endl;
        if (cnt[i].size() > areaLargestCnt){
            if (indexLargestCnt != -1){
                index2ndLargestCnt = indexLargestCnt;
                area2ndLargestCnt = areaLargestCnt;
            }
            indexLargestCnt = i;
            areaLargestCnt = cnt[i].size();
        }else if(cnt[i].size() > area2ndLargestCnt){
            index2ndLargestCnt = i;
            area2ndLargestCnt = cnt[i].size();
        }
    }
    
    // Finding descriptors ----------------------------------------------------------------------------------------------------
    std::vector<cv::Rect> boundRect(cnt.size());
    std::vector<cv::RotatedRect> boundMinArea(cnt.size());
    cv::Point2f rotatedRectPoints_aux[4];
    std::vector<cv::RotatedRect> minEllipse(cnt.size());    // vamos a comparar la relación entre los ejes mayor y menor [didn't yet]
    std::vector<cv::Vec4f> lineCnt(cnt.size());
    std::vector<cv::Moments> mu(cnt.size());
    std::vector<double[7]> huMo(cnt.size());
    
    //Calculating Hu Moments for each contour
    for (int i=0; i<cnt.size(); i++){
        mu[i] = cv::moments(cnt[i]);
        cv::HuMoments(mu[i], huMo[i]);
    }

    //Calculating more descriptors
    for (int i=0; i<cnt.size(); i++){
        boundRect[i] = cv::boundingRect(cnt[i]);
        boundMinArea[i] = cv::minAreaRect(cnt[i]);
        cv::fitLine(cnt[i], lineCnt[i], cv::DIST_L2, 0, 0.01, 0.01);        // L2 distance, more computationally expensive
    }

    // Find info about the largest line ------------------------------------------------------------------------------------------
    if (indexLargestCnt != -1){                                                     // There is a large contour
        // std::cout << "There IS a largest candidate." << std::endl;
        if (index2ndLargestCnt != -1){                                              // There is a 2nd large contour
            if (boundRect[indexLargestCnt].x < boundRect[index2ndLargestCnt].x){    // Si el más grande empieza más a la izquierda
                                                                                    // Vamos por el carril izquierdo, probablemente
                leftLineRegion.insert(leftLineRegion.begin(), cnt[indexLargestCnt]);
                leftLineIndex = indexLargestCnt;
                // std::cout << std::to_string(boundRect[indexLargestCnt].x) << " < " << boundRect[index2ndLargestCnt].x << std::endl;
                rightLineRegion.insert(rightLineRegion.begin(), cnt[index2ndLargestCnt]);
                rightLineIndex = index2ndLargestCnt;
            }else{                                                                  // Vamos por el carril derecho, probablemente
                rightLineRegion.insert(rightLineRegion.begin(), cnt[indexLargestCnt]);
                rightLineIndex = indexLargestCnt;
                // std::cout << std::to_string(boundRect[indexLargestCnt].x) << " > " << boundRect[index2ndLargestCnt].x << std::endl;
                leftLineRegion.insert(leftLineRegion.begin(), cnt[index2ndLargestCnt]);
                leftLineIndex = index2ndLargestCnt;
            }   // Usemos la información de las líneas centrales para asignar la otra línea
        }else{                          // There is no second line
            boundMinArea[indexLargestCnt].points(rotatedRectPoints_aux);
            if( rotatedRectPoints_aux[3].x < binary_eagle.size().width/2){  //Using the bottom-right coordinates point of the minimumRectangleArea to classify them
                leftLineRegion.insert(leftLineRegion.begin(), cnt[indexLargestCnt]);
                leftLineIndex = indexLargestCnt;
                std::cout << std::to_string(rotatedRectPoints_aux[3].x) << " < " << std::to_string(binary_eagle.size().width/2) << std::endl;
            }else{
                rightLineRegion.insert(rightLineRegion.begin(), cnt[indexLargestCnt]);
                rightLineIndex = indexLargestCnt;
                std::cout << std::to_string(rotatedRectPoints_aux[3].x) << " > " << std::to_string(binary_eagle.size().width/2) << std::endl;
            }
        }
    }else{
        std::cout << "There is no largest candidate. :(" << std::endl; // haven't happened yet :D
    }

    //Let's find center lines -------------------------------------------------------------------------------------------------------------------------
    // lineCnt[i]: vector of 4 components (vx,vy,x0,y0) corresponding to the fitting line of the 'i' contour  
    int minDst, maxDst;
    std::vector<int> centerCandidateIndex;
    // Calculating line description of the left and right lines ('m...' is slope and 'b...' is intercept)
    double mLeft, mRight, bLeft, bRight, x_test;        
    if (rightLineRegion.size() > 0 && lineCnt[rightLineIndex][0] != 0){
        mRight = lineCnt[rightLineIndex][1]/lineCnt[rightLineIndex][0];
        bRight = lineCnt[rightLineIndex][3] - mRight*lineCnt[rightLineIndex][2];
    }
    if (leftLineRegion.size() > 0 && lineCnt[leftLineIndex][0] != 0){
        mLeft = lineCnt[leftLineIndex][1]/lineCnt[leftLineIndex][0];
        bLeft = lineCnt[leftLineIndex][3] - mLeft*lineCnt[leftLineIndex][2];
    }
    //Iterate over all contours
    for (int i=0; i<cnt.size(); i++){
        // std::cout << "Lines indexes: left " << std::to_string(leftLineIndex) << " right " << std::to_string(rightLineIndex) << std::endl;
        if (i == leftLineIndex || i == rightLineIndex){     
            // Do nothing, this loop is for the center lines
        }else{
            // Calculate the 4 vertices of the minimum bounding box and assign to 'rotatedRectPoints_aux'
            boundMinArea[i].points(rotatedRectPoints_aux);
            // Calculate the length of the shortest (minDst) and longest (maxDst) side
            minDst = std::min(cv::norm(rotatedRectPoints_aux[0]-rotatedRectPoints_aux[1]), cv::norm(rotatedRectPoints_aux[1]-rotatedRectPoints_aux[2]));
            maxDst = std::max(cv::norm(rotatedRectPoints_aux[0]-rotatedRectPoints_aux[1]), cv::norm(rotatedRectPoints_aux[1]-rotatedRectPoints_aux[2]));
            cv::putText(result, std::to_string(minDst) + " " + std::to_string(maxDst), cv::Point(boundRect[i].tl().x-15, boundRect[i].br().y+15), 
                        cv::FONT_HERSHEY_COMPLEX_SMALL , 0.7, CV_RGB(255,255,255), 1, cv::LINE_8, false);

            // Check about the width of the minAre bounding box and its relative position
            if (minDst < MAX_WIDTH_CENTER_LINE && maxDst < MAX_LENGTH_CENTER_LINE){
                if (leftLineRegion.size() > 0 && rightLineRegion.size() == 0){          // There is only a left line
                    if (lineCnt[leftLineIndex][0] != 0){                                // la pendiente no es infinita
                        x_test = (boundRect[i].y-bLeft)/mLeft;
                        if (boundRect[i].x > x_test){
                            cv::putText(result, "center ll", cv::Point(boundRect[i].tl().x, boundRect[i].br().y+25), 
                                        cv::FONT_HERSHEY_COMPLEX_SMALL , 0.7, CV_RGB(255,255,255), 1, cv::LINE_8, false);
                            centerCandidateIndex.insert(centerCandidateIndex.begin(), i);
                            centerLinesRegion.insert(centerLinesRegion.begin(), cnt[i]);
                        }
                    }else{  // comparar simplemente con la coordenada en x porque la línea es vertical
                        if (boundRect[i].x > boundRect[leftLineIndex].tl().x){
                            cv::putText(result, "center ll", cv::Point(boundRect[i].tl().x, boundRect[i].br().y+25), 
                                        cv::FONT_HERSHEY_COMPLEX_SMALL , 0.7, CV_RGB(255,255,255), 1, cv::LINE_8, false);
                            centerCandidateIndex.insert(centerCandidateIndex.begin(), i);
                            centerLinesRegion.insert(centerLinesRegion.begin(), cnt[i]);
                        }
                    }
                } else if (rightLineRegion.size() > 0 && leftLineRegion.size() == 0){   // There is only a right line
                    if (lineCnt[rightLineIndex][0] != 0){
                        x_test = (boundRect[i].y-bRight)/mRight;    //punto sobre la recta derecha para comparar
                        if (boundRect[i].x < x_test) {
                            cv::putText(result, "center rl", cv::Point(boundRect[i].tl().x, boundRect[i].br().y+25), 
                                        cv::FONT_HERSHEY_COMPLEX_SMALL , 0.7, CV_RGB(255,255,255), 1, cv::LINE_8, false);
                            centerCandidateIndex.insert(centerCandidateIndex.begin(), i);
                            centerLinesRegion.insert(centerLinesRegion.begin(), cnt[i]);
                        }
                    }else{                              // la pendiente es infinita
                        if (boundRect[i].x < boundRect[rightLineIndex].x){
                            cv::putText(result, "center rl", cv::Point(boundRect[i].tl().x, boundRect[i].br().y+25), 
                                        cv::FONT_HERSHEY_COMPLEX_SMALL , 0.7, CV_RGB(255,255,255), 1, cv::LINE_8, false);
                            centerCandidateIndex.insert(centerCandidateIndex.begin(), i);
                            centerLinesRegion.insert(centerLinesRegion.begin(), cnt[i]);
                        }
                    }
                } else if (leftLineRegion.size() > 0 && rightLineRegion.size() > 0){    // There are two lines, left and right
                    if (lineCnt[leftLineIndex][0] != 0){                                // la pendiente no es infinita
                        x_test = (boundRect[i].y-bLeft)/mLeft;
                        if (boundRect[i].x > x_test){
                            if (lineCnt[rightLineIndex][0] != 0){
                                x_test = (boundRect[i].y-bRight)/mRight;    //punto sobre la recta derecha para comparar
                                if (boundRect[i].x < x_test) {
                                    cv::putText(result, "center ll rl", cv::Point(boundRect[i].tl().x, boundRect[i].br().y+25), 
                                                cv::FONT_HERSHEY_COMPLEX_SMALL , 0.7, CV_RGB(255,255,255), 1, cv::LINE_8, false);
                                    centerCandidateIndex.insert(centerCandidateIndex.begin(), i);
                                    centerLinesRegion.insert(centerLinesRegion.begin(), cnt[i]);
                                }
                            }else{  // la pendiente es infinita
                                if (boundRect[i].x < boundRect[rightLineIndex].x) {
                                    cv::putText(result, "center ll rl", cv::Point(boundRect[i].tl().x, boundRect[i].br().y+25), 
                                                cv::FONT_HERSHEY_COMPLEX_SMALL , 0.7, CV_RGB(255,255,255), 1, cv::LINE_8, false);
                                    centerCandidateIndex.insert(centerCandidateIndex.begin(), i);
                                    centerLinesRegion.insert(centerLinesRegion.begin(), cnt[i]);
                                }
                            }
                        }
                    }else{      // la pendiente es infinita
                        if (boundRect[i].x > boundRect[leftLineIndex].x){
                            if (lineCnt[rightLineIndex][0] != 0){           // pendiente no infinita
                                x_test = (boundRect[i].y-bRight)/mRight;    //punto sobre la recta derecha para comparar
                                if (boundRect[i].x < x_test) {
                                    cv::putText(result, "center ll rl", cv::Point(boundRect[i].tl().x, boundRect[i].br().y+25), 
                                                cv::FONT_HERSHEY_COMPLEX_SMALL , 0.7, CV_RGB(255,255,255), 1, cv::LINE_8, false);
                                    centerCandidateIndex.insert(centerCandidateIndex.begin(), i);
                                    centerLinesRegion.insert(centerLinesRegion.begin(), cnt[i]);
                                }
                            }else{  // la pendiente es infinita
                                if (boundRect[i].x < boundRect[rightLineIndex].x) {
                                    cv::putText(result, "center ll rl", cv::Point(boundRect[i].tl().x, boundRect[i].br().y+25), 
                                                cv::FONT_HERSHEY_COMPLEX_SMALL , 0.7, CV_RGB(255,255,255), 1, cv::LINE_8, false);
                                    centerCandidateIndex.insert(centerCandidateIndex.begin(), i);
                                    centerLinesRegion.insert(centerLinesRegion.begin(), cnt[i]);
                                }
                            }
                        }
                    }
                    
                }else{
                    cv::putText(result, "2b inspected", cv::Point(boundRect[i].tl().x-20, boundRect[i].br().y+25), 
                            cv::FONT_HERSHEY_COMPLEX_SMALL , 0.7, CV_RGB(255,255,255), 1, cv::LINE_8, false);
                }
            }
        }
    }

    for (int i=0; i<leftLineRegion.size(); i++){       // Drawing contours of the left line
        cv::drawContours(result, leftLineRegion, i, cv::Scalar(0,255,0), cv::FILLED, cv::LINE_8);
    }
    for (int i=0; i<rightLineRegion.size(); i++){       // Drawing contours of the right line
        cv::drawContours(result, rightLineRegion, i, cv::Scalar(0,0,255), cv::FILLED, cv::LINE_8);
    }
    for (int i=0; i<centerLinesRegion.size(); i++){       // Drawing contours of the center lines
        cv::drawContours(result, centerLinesRegion, i, cv::Scalar(255,255,0), cv::FILLED, cv::LINE_8);
    }

    // Drawing contours and rectangles -----------------------------------------------------------------------
    for (int i=0; i<cnt.size(); i++){
        cv::rectangle(result, boundRect[i].tl(), boundRect[i].br(), cv::Scalar(255,255,255), 1, cv::LINE_8);
        boundMinArea[i].points(rotatedRectPoints_aux);
        for (int j=0; j<4; j++) cv::line(result, rotatedRectPoints_aux[j], rotatedRectPoints_aux[(j+1)%4], cv::Scalar(150,150,0));

        if (i==rightLineIndex || i==leftLineIndex){     // Checking parking spots recognition
            for (int hu=0; hu<7; hu++){
                cv::putText(result, std::to_string(huMo[i][hu]), cv::Point(boundRect[i].x, boundRect[i].y+15*hu+50),
                            cv::FONT_HERSHEY_COMPLEX_SMALL , 0.8, CV_RGB(255,255,255), 1, cv::LINE_8, false);
            }
            if (huMo[i][0]+huMo[i][1]+huMo[i][2]> 2 && huMo[i][0]+huMo[i][1]+huMo[i][2] <5){
                if (huMo[i][3]+huMo[i][4]+huMo[i][5] < 3 && huMo[i][3]+huMo[i][4]+huMo[i][5] > 0.3){
                    cv::putText(result, "PARKING!!!!!!!!", cv::Point(boundRect[i].tl().x, boundRect[i].br().y+30),
                                cv::FONT_HERSHEY_COMPLEX_SMALL , 0.8, CV_RGB(255,0,255), 1, cv::LINE_8, false);
                }
            }
        }   
    }

    cv::imshow("Clasification logic (center w/ respect to side lines)", result);
    // cv::waitKey(0);
    // std::cout << std::endl;
    std::vector<std::vector<std::vector<cv::Point>>> output;
    output.insert(output.begin(), leftLineRegion);
    output.insert(output.begin(), centerLinesRegion);
    output.insert(output.begin(), rightLineRegion);
    return output;
}

std::vector<bool> filterConnectedCenterLines(std::vector<std::vector<cv::Point>> candidatesContours,std::vector<bool> isCandidate){
    bool check1 , check2;
    std::vector<bool> output(isCandidate.size(),false);
    for (size_t i = 0; i < isCandidate.size(); i++)
    {
        if(isCandidate[i]){
            for (size_t j = 0; j < isCandidate.size(); j++)
            {
                if(isCandidate[j] && i != j){
                    //std::cout<<i<<" with "<<j;
                    check1 = isAligned(candidatesContours[i],candidatesContours[j]);
                    check2 = isAligned(candidatesContours[j],candidatesContours[i]);
                    if(check1 and check2){
                        output[i] = true;
                        output[j] = true;
                        //std::cout<<i<<" positive "<<j;
                    }
                    //std::cout<<"\n";
                }   
            } 
        }
    }
    return output;
}

bool isAligned(std::vector<cv::Point> area1, std::vector<cv::Point> area2){
    const float TOLERANCE = 30; 
    const float TOLERANCE_HIT = 10;
    const bool slope_mode = true;
    cv::Vec4f output1, output2;
    cv::fitLine(area1,output1,cv::DIST_L2, 0, 0.01, 0.01);
    cv::fitLine(area2,output2,cv::DIST_L2, 0, 0.01, 0.01);

    //Calculate distance between the two centers and reject very close/far lines
    float distance = cv::norm(cv::Point2f(output2[2],output2[3]) - cv::Point2f(output1[2],output1[3]));
    if(distance < 100 || distance > 175){
        return false;
    }

    //Avoid divisions by 0
    if(output1[1] == 0) output1[1] = 0.001;
    if(output2[3] == output1[3]) output1[3] += 0.001;

    if(output1[0] == 0) output1[0] = 0.001;
    if(output2[2] == output1[2]) output1[2] += 0.001;

    //Calculate slopes and projections along the vector for each axis
    float slope_y = output1[0] / output1[1];
    float x_test = (output2[3] - output1[3]) * slope_y + output1[2];
    float dx = std::abs(x_test - output2[2]);

    float slope_x = output1[1] / output1[0];
    float y_test = (output2[2] - output1[2]) * slope_x + output1[3];
    float dy = std::abs(y_test - output2[3]);
    //std::cout<<x_test<<" vs. "<<output2[2]<<". Diff: "<<dy<<". \t";

    //Check if the projected vector hits the bounding box of the second line
    cv::Rect bb = cv::boundingRect(area2);
    float v1_x= bb.br().x;
    float v1_y= bb.br().y;
    float v2_x= bb.tl().x;
    float v2_y= bb.tl().y;

    float hit = (v1_x - output1[2]) * slope_x + output1[3];
    if((hit >= v1_y - TOLERANCE_HIT && hit <= v2_y + TOLERANCE_HIT) || (hit >= v2_y - TOLERANCE_HIT && hit <= v1_y + TOLERANCE_HIT))    return true;
    hit = (v2_x - output1[2]) * slope_x + output1[3];
    if((hit >= v1_y - TOLERANCE_HIT && hit <= v2_y + TOLERANCE_HIT) || (hit >= v2_y - TOLERANCE_HIT && hit <= v1_y + TOLERANCE_HIT))    return true;
    hit = (v1_y - output1[3]) * slope_y + output1[2];
    if((hit >= v1_x - TOLERANCE_HIT && hit <= v2_x + TOLERANCE_HIT) || (hit >= v2_x - TOLERANCE_HIT && hit <= v1_x + TOLERANCE_HIT))    return true;
    hit = (v2_y - output1[3]) * slope_y + output1[2];
    if((hit >= v1_x - TOLERANCE_HIT && hit <= v2_x + TOLERANCE_HIT) || (hit >= v2_x - TOLERANCE_HIT && hit <= v1_x + TOLERANCE_HIT))    return true;


    //Compare the projection and the position of the second line
    if( dy < TOLERANCE || dx < TOLERANCE){
        //std::cout<<y_test<<" vs. "<<output2[3]<<"\n";
        return true;
    } 
    return false;
}

float getPositionAtBottom(std::vector<cv::Point> line){
    //const int HEIGHT_IMAGE = 480;
    //cv::Vec4f output;
    //cv::fitLine(line,output,cv::DIST_L2, 0, 0.01, 0.01);
    //if(output[1] == 0)  output[1]+=0.001;
    //float x_test = output[2] + (HEIGHT_IMAGE - output[3])*(output[0]/output[1]);
    //return x_test;

    return getMaxYPoint(line).x;
}

cv::Point getMaxYPoint(std::vector<cv::Point> region){
    auto max_x_point_iterator = std::max_element(region.begin(), region.end(),
        [](const cv::Point& p1, const cv::Point& p2) {
            return p1.y < p2.y;
        });
    cv::Point output = *max_x_point_iterator;
    return output;
} 

// if (index2ndLargestCnt != -1){                                              // There is a 2nd large contour
//         // std::cout << "Actually, there are two! :D" << std::endl;
//         // cv::putText(result, "There are two big groups (line 119)", cv::Point(50, 700), 
//         //                 cv::FONT_HERSHEY_COMPLEX_SMALL , 3, CV_RGB(255,255,255), 1, cv::LINE_8, false);
//         if (boundRect[indexLargestCnt].x < boundRect[index2ndLargestCnt].x){    // Si el más grande empieza más a la izquierda
//                                                                                 // Vamos por el carril izquierdo, probablemente
//             leftLineRegion.insert(leftLineRegion.begin(), cnt[indexLargestCnt]);
//             leftLineIndex = indexLargestCnt;
//             // std::cout << std::to_string(boundRect[indexLargestCnt].x) << " < " << boundRect[index2ndLargestCnt].x << std::endl;
//             rightLineRegion.insert(rightLineRegion.begin(), cnt[index2ndLargestCnt]);
//             rightLineIndex = index2ndLargestCnt;
//         }else{                                                                  // Vamos por el carril derecho, probablemente
//             rightLineRegion.insert(rightLineRegion.begin(), cnt[indexLargestCnt]);
//             rightLineIndex = indexLargestCnt;
//             // std::cout << std::to_string(boundRect[indexLargestCnt].x) << " > " << boundRect[index2ndLargestCnt].x << std::endl;
//             leftLineRegion.insert(leftLineRegion.begin(), cnt[index2ndLargestCnt]);
//             leftLineIndex = index2ndLargestCnt;
//         }   // Usemos la información de las líneas centrales para asignar la otra línea
//     }else{                          // There is no second line
//         boundMinArea[indexLargestCnt].points(rotatedRectPoints_aux);
//         if( rotatedRectPoints_aux[3].x < binary_eagle.size().width/2){  //Using the bottom-right coordinates point of the minimumRectangleArea to classify them
//             leftLineRegion.insert(leftLineRegion.begin(), cnt[indexLargestCnt]);
//             leftLineIndex = indexLargestCnt;
//             std::cout << std::to_string(rotatedRectPoints_aux[3].x) << " < " << std::to_string(binary_eagle.size().width/2) << std::endl;
//         }else{
//             rightLineRegion.insert(rightLineRegion.begin(), cnt[indexLargestCnt]);
//             rightLineIndex = indexLargestCnt;
//             std::cout << std::to_string(rotatedRectPoints_aux[3].x) << " > " << std::to_string(binary_eagle.size().width/2) << std::endl;
//         }
//     }