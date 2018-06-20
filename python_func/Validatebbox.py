#----------------------------------------------------------------------
# this python file is used to validate the bounding box of dataset3
#----------------------------------------------------------------------
import cv2
import pickle
import numpy as np 

video_path = '/home/oliver/data/car/Dense/jan28.avi'
cap = cv2.VideoCapture(video_path)
flag = 0

fin = open('./car_bbox.bin','rb')
car_info = pickle.load(fin)


while(cap.isOpened()):
    print flag
    cap.set(cv2.CAP_PROP_POS_FRAMES, flag)
    ret, frame = cap.read() 
    if ret == False:
        break
    current_frame_bbox = car_info[str(flag)]
    for i in range(current_frame_bbox.shape[0]):
        x1 = int(current_frame_bbox[i, 0])
        y1 = int(current_frame_bbox[i, 1])
        x2 = int(current_frame_bbox[i, 0] + current_frame_bbox[i, 2])
        y2 = int(current_frame_bbox[i, 1] + current_frame_bbox[i, 3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    flag = flag + 1

cap.release()
cv2.destroyAllWindows()


