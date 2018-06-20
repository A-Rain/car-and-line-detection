#------------------------------------------------------------------------
# this python file is used to generate the retrain img data so that we
# we could find and use those hard example to make the model more accurate
# I use ProData6 as the retrain data list
#------------------------------------------------------------------------
import numpy as np
import os

data_path = '/home/oliver/data/ProData6/Annotations/'
# data_path = '/home/oliver/data/ProData2/'
imglist = []

files = sorted(os.listdir(data_path))
for imgname in files:
    imglist.append(imgname.split('.')[0])

imglist = np.array(imglist, dtype=np.str)
imglist = np.reshape(imglist, (-1, 1))
np.savetxt('Retrain.txt', imglist, fmt='%s')
print 'build retrainlist successfully!'
