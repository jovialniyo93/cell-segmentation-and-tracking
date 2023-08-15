import cv2
import os
import shutil
import numpy as np

def useAreaFilter(img,area_size):
    if len(np.unique(img))!=2:
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, markers = cv2.connectedComponents(img)
    list=np.unique(markers)[1:]
    for label in list:
        mask = (markers == label) * 1
        area=np.sum(mask)
        if area<area_size:
            markers-=mask*label
    markers=(markers>0)*255
    markers=markers.astype(np.uint8)
    return markers


def createFolder(path,clean=False):
    if not os.path.isdir(path):
        os.mkdir(path)
        print("\t\t{} has been created.".format(path))
    else:
        if clean:
            file_list = os.listdir(path)
            for file in file_list:
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            print("\t\t{} already exist,files in it will be cleaned!".format(path))
        else:
            print("\t\t{} already exist.".format(path))

def deleteFile(path):
    file_list=os.listdir(path)
    for file in file_list:
        file_path=os.path.join(path,file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    print("\t\tFiles in {} has been deleted.".format(path))

def copyFile(source_path,target_path):
    source_file_list=os.listdir(source_path)
    print("\t\tFiles in {} has been copied to {}.".format(source_path,target_path))
    for file_name in source_file_list:
        source_file=os.path.join(source_path,file_name)
        shutil.copy(source_file,target_path)