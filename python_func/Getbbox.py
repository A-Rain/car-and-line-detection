import cv2
import numpy as np
import pickle

vedio_path = '/home/oliver/data/car/Dense/pos_annot.dat'
store_path = './car_bbox.bin'

bbox = {}

with open(vedio_path) as f:
    annot=f.readline()
    while annot != '':
        info = annot.split()
        # print info
        ID = info[0]
        CarNum = int(info[1])
        # xmin, ymin, height, width
        CarInfo = np.zeros((CarNum, 4))
        k = 0
        i = 2
        for k in range(CarNum):
            CarInfo[k, 0] = int(info[i + k * 4])
            CarInfo[k, 1] = int(info[i + 1 + k * 4])
            CarInfo[k, 2] = int(info[i + 2 + k * 4])
            CarInfo[k, 3] = int(info[i + 3 + k * 4])
        bbox[ID] = CarInfo
        annot=f.readline()
    
f.close()

fin = open(store_path, 'wb')
pickle.dump(bbox, fin)
fin.close()
