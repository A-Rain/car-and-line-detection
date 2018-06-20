import cv2
import pickle
import numpy as np 
import os

def ImgID(No):
    if No<10:
        return 'IMG_000' + str(No) + '.jpg'
    elif No>=10 and No<100:
        return 'IMG_00' + str(No) + '.jpg'
    elif No>=100 and No<1000:
        return 'IMG_0' + str(No) + '.jpg'
    else:
        return 'IMG_' + str(No) + '.jpg'

video_path = '/home/oliver/data/car/Dense/jan28.avi'
writeing_img_path = '/home/oliver/data/ProD' 
cap = cv2.VideoCapture(video_path)
flag = 0



while(cap.isOpened()):
    print flag
    cap.set(cv2.CAP_PROP_POS_FRAMES, flag)
    ret, frame = cap.read() 
    if ret == False:
        break
    curr_path = os.path.join(writeing_img_path, ImgID(flag))
    cv2.imwrite(curr_path, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    flag = flag + 1

cap.release()
cv2.destroyAllWindows()

