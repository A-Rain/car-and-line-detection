#----------------------------------------------------------------------------
# this python file is used to build positive imagelist for retraining, we use
# positive image from dataset3 ground truth box and dataset2 positive image
#----------------------------------------------------------------------------
import cv2
import numpy as np
import os


data_path = '/home/oliver/data/ProData3/'
data_writing_path = '/home/oliver/data/ProData7/'
path_vechile = '/home/oliver/data/car_noncar_data/vehicles'
ID = 0
X = []


    
def ImgName(Noo):
    if Noo<10:
        return 'IMG_0000' + str(Noo) + '.jpg'
    elif Noo>=10 and Noo<100:
        return 'IMG_000' + str(Noo) + '.jpg'
    elif Noo>=100 and Noo<1000:
        return 'IMG_00' + str(Noo) + '.jpg'
    elif Noo>=1000 and Noo<10000:
        return 'IMG_0' + str(Noo) + '.jpg'
    else:
        return 'IMG_' + str(Noo) + '.jpg'


dirsss = os.listdir(path_vechile)
for dirss in dirsss:
    dirss_path = os.path.join(path_vechile, dirss)
    dirs = os.listdir(dirss_path)
    if ID == 5200:
        break
    for _file in dirs:       
        ID = ID + 1         
        if ID <= 4000:
           continue
        if ID == 5200:
            break
        path = os.path.join(dirss_path, _file)
        pic = cv2.imread(path)
        cv2.imwrite(data_writing_path + ImgName(ID), pic)
        X.append(ImgName(ID))
        if ID%1000 == 0:
           print ID

dirs = sorted(os.listdir(data_path))
for _file in dirs:
    img_path = os.path.join(data_path, _file)
    img = cv2.imread(img_path)
    cv2.imwrite(data_writing_path + ImgName(ID), img)
    X.append(ImgName(ID))
    ID = ID + 1
    if ID%1000 == 0:
        print ID

X = np.array(X, dtype=np.str)
X = np.reshape(X, (-1, 1))
Y = np.ones((X.shape[0], 1), dtype=np.str)
X = np.hstack((X, Y))

np.savetxt("retrain_pos.txt", X, fmt='%s')







