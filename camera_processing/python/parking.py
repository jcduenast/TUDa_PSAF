import cv2
import numpy as np
from matplotlib import pyplot as plt
import csv
import time
from math import copysign,log10

def parseToNumber(object):
    output = [0,0,0,0,0,0,0]
    for i in range(0,7):
        output[i] = float(object[i])
    return output

def getDistance(object1, object2):
    diff = [0,0,0,0,0,0,0]
    for i in range(0,7):
        diff[i] = abs(object1[i] - object2[i])
    return sum(diff)

path = '/home/daniel/Documentos/TU/PSAF/TUDa_PSAF/camera_processing/'
folders = ['curveRight/parking/','curveRight/curve/', 'curveLeft/parking/', 'curveLeft/curve/']

park = [-0.2143883199819339, 0.42935117420838054, 0.15538505565471847, 1.5660314323545717, 2.663938705869138, -1.886375915742787, 2.515469319909715]
curve = [-0.14392678791516805, -0.26672432011402, 0.7350189406203387, 1.5010662464830626, 3.3754487242627413, -1.9792000856518963, 2.6258826141919327]
with open(path+'huMoments.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        if(row[8].count('curveRight-') > 0):
            d = getDistance(parseToNumber(row),park)
            e = getDistance(parseToNumber(row),curve)
            if(d<e and d<5):
                print('park',row[8], row[7], d)
            elif(e<d and e<5):
                print('curve',row[8], row[7], e)
            


