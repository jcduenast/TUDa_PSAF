import cv2
import numpy as np



def getLaneSegmentation(raw_cam_color):
    gray = cv2.cvtColor(raw_cam_color, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0, 0, cv2.BORDER_DEFAULT)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=101, C=-50)
    binary_eagle = get_eagle_view(binary, 2)

    leftLineRegion = []
    centerLinesRegion = []
    rightLineRegion = []
    leftLineIndex = None
    rightLineIndex = None

    # Contour detection
    cntAll, _ = cv2.findContours(binary_eagle, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    height, width = binary_eagle.size()
    result = np.zeros((height,width,3), np.uint8)

    # Draw all contours in grey, will be overwritten when classified
    for i in range(len(cntAll)):
        cv2.drawContours(result, cntAll, i, (100, 100, 100), thickness=cv2.FILLED, lineType=cv2.LINE_8)

    # Delete small contours
    cnt = [contour for contour in cntAll if len(contour) > 100]

    # Finding the contour (cnt) with the largest area to tag them as either left or right lines
    indexLargestCnt = -1
    index2ndLargestCnt = -1
    areaLargestCnt = 600
    area2ndLargestCnt = 600  # Side lines should be at least 600 big, normally around +900

    for i in range(len(cnt)):
        if len(cnt[i]) > areaLargestCnt:
            if indexLargestCnt != -1:
                index2ndLargestCnt = indexLargestCnt
                area2ndLargestCnt = areaLargestCnt
            indexLargestCnt = i
            areaLargestCnt = len(cnt[i])
        elif len(cnt[i]) > area2ndLargestCnt:
            index2ndLargestCnt = i
            area2ndLargestCnt = len(cnt[i])
    
    mu = [cv2.moments(contour) for contour in cnt]
    huMo = [cv2.HuMoments(moment) for moment in mu]

    boundRect = [cv2.boundingRect(contour) for contour in cnt]
    boundMinArea = [cv2.minAreaRect(contour) for contour in cnt]

    if indexLargestCnt != -1:  # There is a large contour
        if index2ndLargestCnt != -1:  # There is a 2nd large contour
            if boundRect[indexLargestCnt][0] < boundRect[index2ndLargestCnt][0]:    # [x, y, w, h]
                # If the largest starts more to the left, probably in the left lane
                leftLineRegion.insert(0, cnt[indexLargestCnt])
                leftLineIndex = indexLargestCnt

                rightLineRegion.insert(0, cnt[index2ndLargestCnt])
                rightLineIndex = index2ndLargestCnt
            else:
                # Probably in the right lane
                rightLineRegion.insert(0, cnt[indexLargestCnt])
                rightLineIndex = indexLargestCnt

                leftLineRegion.insert(0, cnt[index2ndLargestCnt])
                leftLineIndex = index2ndLargestCnt
        else:  # There is no second line
            boundMinArea[indexLargestCnt].points(rotatedRectPoints_aux)
            if rotatedRectPoints_aux[3][0] < binary_eagle.shape[1] / 2:
                # Using the bottom-right coordinates point of the minimumRectangleArea to classify them
                leftLineRegion.insert(0, cnt[indexLargestCnt])
                leftLineIndex = indexLargestCnt
            else:
                rightLineRegion.insert(0, cnt[indexLargestCnt])
                rightLineIndex = indexLargestCnt
    else:
        print("There is no largest candidate. :(")    
    return result


def get_eagle_view(img_in, mode):
    homography_data_psaf1000_wider = np.array([-1.292647040542012, -2.9644706663624953, 722.7746234622502,
                                               0.014571325976530337, -6.298569281531482, 1220.7592525637147,
                                               2.7245596705395877e-05, -0.009299076219158845, 1.0], dtype=np.float64).reshape((3, 3))
    if mode == 2:
        homography = np.array(homography_data_psaf1000_wider, dtype=np.float64).reshape((3, 3))
        projected = cv2.warpPerspective(img_in, homography, (640, 640))
        return projected
    else:
        print("No other mode supported. Refer to original version in C++")
    return
    