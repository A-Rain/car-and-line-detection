# ---------------------------------------------------------------------
# this python file is used to store bnd box in .bin and write xml file
# and validate it for dataset4
# ---------------------------------------------------------------------
import cv2
import numpy as np
import pickle
import os
from xml.dom import minidom
import xml.etree.ElementTree as ET

data_path = '/home/oliver/data/ProData6/JEPGImages/'
bin_file = 'car_bbox2.bin'
annotation = '/home/oliver/data/ProData6/Annotations/'
ratio_x = 448./1920.
ratio_y = 448./1200.

def Parsexml(filename):
    tree = ET.ElementTree(file=filename)
    imgname = tree.findtext('filename')
    print imgname
    object_num = 0
    for elem in tree.iter(tag='object'):
        object_num = object_num + 1
    record = np.zeros([object_num, 5], dtype=np.float)

    object_num = 0
    for elem in tree.iter(tag='object'):
        strs = elem[0].text
        if strs == 'Car':
            record[object_num, 0] = 1
        elif strs == 'Truck':
            record[object_num, 0] = 2
        else:
            record[object_num, 0] = 0
        record[object_num, 0] = 1
        for elem2 in elem.iter(tag='bndbox'):
            record[object_num, 1] = float(elem2.findtext('xmin'))*ratio_x
            record[object_num, 2] = int(elem2.findtext('ymin'))*ratio_y
            record[object_num, 3] = int(elem2.findtext('xmax'))*ratio_x
            record[object_num, 4] = int(elem2.findtext('ymax'))*ratio_y
        object_num = object_num + 1
    return record, imgname

def xmlwriting(info):
    xml = minidom.Document()
    root = xml.createElement('annotation')
    xml.appendChild(root)
    img_name = info[0, 4]
    filename = xml.createElement('filename')
    filename.appendChild(xml.createTextNode(img_name))
    root.appendChild(filename)
    for i in range(info.shape[0]):
        obj = xml.createElement("object")
        name = xml.createElement("name")
        xmin = xml.createElement("xmin")
        xmax = xml.createElement("xmax")
        ymin = xml.createElement("ymin")
        ymax = xml.createElement("ymax")
        bndbox = xml.createElement("bndbox")
        name.appendChild(xml.createTextNode(info[i, 5]))
        xmin.appendChild(xml.createTextNode(info[i, 0]))
        ymin.appendChild(xml.createTextNode(info[i, 1]))
        xmax.appendChild(xml.createTextNode(info[i, 2]))
        ymax.appendChild(xml.createTextNode(info[i, 3]))
        obj.appendChild(name)
        bndbox.appendChild(xmin)
        bndbox.appendChild(ymin)
        bndbox.appendChild(xmax)
        bndbox.appendChild(ymax)
        obj.appendChild(bndbox)
        root.appendChild(obj)
        filename = annotation+img_name.split('.')[0]+'.xml'
        f = open(filename, 'w')
        f.write(xml.toprettyxml(encoding='utf-8'))
        f.close()



if __name__ == '__main__':
# xmin,ymin,xmax,ymax,Frame,Label,Preview URL
# if not yet write xml
    if not os.path.exists(annotation):
        os.mkdir(annotation)
        label_path = os.path.join(data_path, 'label.txt')
        info = np.loadtxt(label_path, delimiter=',', dtype='str')
        info = info[1:, :6]
        count = 0
        i = 0
        while i < info.shape[0]:
            count = 0
            while i + count + 1 < info.shape[0] and info[i + count, 4] == info[i + count + 1, 4]:
                count = count + 1
            bbox4img = info[i:i + count + 1]
            xmlwriting(bbox4img)
            i = i + count + 1
            if i % 1000 == 0:
                print i
    
    imglist = sorted(os.listdir(annotation))
    for annot in imglist:
        annot_path = os.path.join(annotation, annot)
        bbox_info, imgname = Parsexml(annot_path)
        pic_path = os.path.join(data_path, imgname)
        img = cv2.imread(pic_path)
        img = cv2.resize(img, (448, 448))
        # print bbox_info
        for i in range(bbox_info.shape[0]):
            if bbox_info[i][0] != 0:
                cv2.rectangle(img, (int(bbox_info[i][1]), int(bbox_info[i][2])), (int(bbox_info[i][3]), int(bbox_info[i][4])), (0, 255, 0), 2)
        cv2.imshow('img', img)
        cv2.waitKey(100)
