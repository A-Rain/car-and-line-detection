import cv2
import os

im_dir = '/home/oliver/data/ProData6/JEPGImages'

video_dir = '/home/oliver/data/out2.avi'

fps = 15

num = 0

img_size = (1920, 1200)

# fourcc = cv2.cv.CV_FOURCC('M','J','P','G')#opencv2.4
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # opencv3.0
videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

dirs = sorted(os.listdir(im_dir))
for img in dirs:
    num = num + 1
    if num < 1600:
        continue
    im_name = os.path.join(im_dir, img)
    frame = cv2.imread(im_name)
    videoWriter.write(frame)
    print im_name
    if num == 2500:
        break

videoWriter.release()
print 'finish'
