#------------------------------------------------------------------------
# this python file is used to generate the training data and test data.
# to make it more accurate, I choose about 4000 car data and 12000 
# non car data and split them into training and test. I use two datasets.
#------------------------------------------------------------------------
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

No = 0
X = []
Y = []

# datset1 path
path_vechile = '/home/oliver/data/car_noncar_data/vehicles'
path_non_vechile = '/home/oliver/data/car_noncar_data/non-vehicles'
# dataset2 path
path_non_vechile2 = '/home/oliver/data/ProData4/'

def ImgName(Noo):
    if Noo<10:
        return 'IMG_0000' + str(Noo) + '.png'
    elif Noo>=10 and No<100:
        return 'IMG_000' + str(Noo) + '.png'
    elif Noo>=100 and Noo<1000:
        return 'IMG_00' + str(Noo) + '.png'
    elif Noo>=1000 and Noo<10000:
        return 'IMG_0' + str(Noo) + '.png'
    else:
        return 'IMG_' + str(Noo) + '.png'
    

print 'dataset 1 --- vechile'
dirsss = os.listdir(path_vechile)
for dirss in dirsss:
    if No == 4000:
       break
    dirss_path = os.path.join(path_vechile, dirss)
    dirs = os.listdir(dirss_path)   
    for file in dirs:
        path = os.path.join(dirss_path, file)
        pic = cv2.imread(path)
        cv2.imwrite('/home/oliver/data/ProData/' + ImgName(No), pic)
        X.append(ImgName(No))
        Y.append(1)
        No = No + 1         
        if No%1000 == 0:
           print No
        if No == 4000:
           break

print 'dataset 1 --- non vechile' 
dirsss = os.listdir(path_non_vechile)
for dirss in dirsss:
    dirss_path = os.path.join(path_non_vechile, dirss)
    dirs = os.listdir(dirss_path)
    for file in dirs:
        path = os.path.join(dirss_path, file)
        pic = cv2.imread(path)
        cv2.imwrite('/home/oliver/data/ProData/' + ImgName(No), pic)
        X.append(ImgName(No))
        Y.append(0)
        No = No + 1         
        if No%1000 == 0:
           print No

print 'dataset 2 --- non vechile'
dirsss = os.listdir(path_non_vechile2)
for _file in dirsss:
    path = os.path.join(path_non_vechile2, _file)
    pic = cv2.imread(path)
    cv2.imwrite('/home/oliver/data/ProData/' + ImgName(No), pic)
    X.append(ImgName(No))
    Y.append(0)
    No = No + 1         
    if No%1000 == 0:
        print No
    if No == 16000:
        break


            
X = np.array(X, dtype=np.str)
X = np.reshape(X, (-1, 1))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
Y_train = np.array(Y_train).astype(np.str)
Y_train = np.reshape(Y_train, (-1, 1))
Y_test = np.array(Y_test).astype(np.str)
Y_test = np.reshape(Y_test, (-1, 1))
print X_train.shape
print Y_train.shape
trainlist = np.hstack((X_train, Y_train))
testlist = np.hstack((X_test, Y_test))
np.savetxt("test.txt", testlist, fmt='%s')
np.savetxt("train.txt", trainlist, fmt='%s')
