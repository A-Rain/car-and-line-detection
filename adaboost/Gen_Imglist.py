import os
import cv2
import numpy as np
path = '/home/oliver/data/ProData3/'
relative_path = '../../../data/ProData3'
RoI = ' 1 0 0 '
pos_list = []
neg_list = []

No = 0

path2 = '/home/oliver/data/ProData4/'
relative_path2 = '../../../data/ProData4'

# generate negative dat
dirs = os.listdir(path2)
for img_name in dirs:
    img_path = os.path.join(relative_path2, img_name)
    neg_list.append(img_path)
    if No % 1000 == 0:
        print No
    No = No + 1

neg_list = np.array(neg_list, dtype = np.str)
neg_list = np.reshape(neg_list, (-1, 1))
np.savetxt('neg_info.dat', neg_list, fmt='%s')

exit(0)

# generate postive dat
dirs = os.listdir(path)
for img_name in dirs:
    img_path = os.path.join(relative_path, img_name)
    pic = cv2.imread(img_path)
    (h, w) = pic.shape[:2]
    Region = RoI + str(w) + ' ' + str(h)
    vec = img_path + Region
    pos_list.append(vec)
    if No % 1000 == 0:
        print No
    No = No + 1

pos_list = np.array(pos_list, dtype = np.str)
pos_list = np.reshape(pos_list, (-1, 1))
np.savetxt('pos_info.dat', pos_list, fmt='%s')


