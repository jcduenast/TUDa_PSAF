import cv2
import numpy as np
from matplotlib import pyplot as plt
import csv
import time
from math import copysign,log10

path = '/home/daniel/Documentos/TU/PSAF/TUDa_PSAF/camera_processing/img/'
folders = ['curveRight/parking/','curveRight/curve/', 'curveLeft/parking/', 'curveLeft/curve/']


for folder in folders:

    with open(path+folder+'list.csv', newline='') as csvfile, open(path+folder+'huMoments.csv','w',newline='') as outputfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        csvWriter = csv.writer(outputfile)
        #fig,ax = plt.subplots(figsize=(8,6))
        for row in spamreader:
            name = ', '.join(row)
            if(name=='list.csv'):
                break
            img = cv2.imread(path+folder+name,0)
            #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(img,150,255,0)
            
            contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            #print("Number of Contours detected:",len(contours))
            # Find the moments of first contour
            cnt = contours[0]
            M = cv2.moments(cnt)
            Hm = cv2.HuMoments(M)
            # Log scale hu moments 
            HuMoments = [0,0,0,0,0,0,0]
            for i in range(0,7):
                HuMoments[i] = -1* copysign(1.0, Hm[i][0]) * log10(abs(Hm[i][0]))
            csvWriter.writerow(HuMoments)
