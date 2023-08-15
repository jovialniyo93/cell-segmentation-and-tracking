import cv2
import os
import numpy as np
from track.track_3_period import *
from tools.tool import *

def get_average_list(img_path,track_path,period):
    image_list=os.listdir(img_path)
    image_list.sort()
    track_result_path=os.path.join(track_path,str(period).zfill(2)+"/"+str(period).zfill(2)+"_RES")
    track_list=os.listdir(track_result_path)
    track_list = [name for name in track_list if ".tif" in name]
    track_list.sort()
    if period == 1:
        image_list = image_list[:102]
    if period == 2:
        image_list = image_list[102:143]
    if period == 3:
        image_list = image_list[143:]
    average_whole_list=[]
    for i,name in enumerate(image_list):
        image=cv2.imread(os.path.join(img_path,name),-1)
        image=image[5:741,1:769]
        mask=cv2.imread(os.path.join(track_result_path,track_list[i]),-1)
        label_list=np.unique(mask)[1:]
        average_single_list=[]
        for label in label_list:
            mark=(mask==label)*1
            area=np.sum(mark)
            value=np.sum(mark*image)
            average=value/area
            average_single_list.append(average)
        average_whole_list.append(np.mean(average_single_list))
    return average_whole_list

def get_average(img_path,track_path,period,label,frame):
    image_list = os.listdir(img_path)
    image_list.sort()
    track_result_path = os.path.join(track_path, str(period).zfill(2) + "/" + str(period).zfill(2) + "_RES")
    track_list = os.listdir(track_result_path)
    track_list = [name for name in track_list if ".tif" in name]
    track_list.sort()
    if period == 1:
        image_list = image_list[:102]
    if period == 2:
        image_list = image_list[102:143]
    if period == 3:
        image_list = image_list[143:]
    image=cv2.imread(os.path.join(img_path,image_list[frame]),-1)
    image = image[5:741, 1:769]
    mask=cv2.imread(os.path.join(track_result_path,track_list[frame]),-1)
    mask=(mask==label)*1
    area=np.sum(mask)
    value=np.sum(mask*image)
    return value/area

def get_information(track_path,period,label):
    track_result_path = os.path.join(track_path, str(period).zfill(2) + "/" + str(period).zfill(2) + "_RES")
    record_file=os.path.join(track_result_path,"res_track.txt")
    with open(record_file, "r") as f:
        data = f.readlines()
    lines = [line.strip('\n') for line in data]
    for line in lines:
        line = line.split()
        number = int(line[0])
        if number==label:
            start = int(line[1])
            end = int(line[2])
            parent_number = int(line[3])
            break
    return start,end,parent_number

def haveChild(track_path,period,label):
    track_result_path = os.path.join(track_path, str(period).zfill(2) + "/" + str(period).zfill(2) + "_RES")
    record_file = os.path.join(track_result_path, "res_track.txt")
    with open(record_file, "r") as f:
        data = f.readlines()
    lines = [line.strip('\n') for line in data]
    for line in lines:
        line = line.split()
        parent_number = int(line[3])
        if parent_number==label:
            return True
    return False

def locationNotInRange(track_result_path,label,end):
    (x,y)=get_center(end,label,track_result_path)
    if x in range(14,754) and y in range(14,722):
        return False
    return True

def ifPixelValue(img_path,track_path,period,number,start,end,average_list):
    count=0
    for i in range(start, end + 1, 1):
        average = get_average(img_path, track_path, period, number, i)
        if average < (2 * average_list[i]):
            count += 1
    if count>1:
        return True
    return False
def reduceFP_with_track(img_path,track_path,period):
    FP_path=os.path.join(track_path,str(period).zfill(2) + "/""remove_FP")
    createFolder(FP_path,clean=True)
    image_list = os.listdir(img_path)
    image_list.sort()
    track_result_path = os.path.join(track_path, str(period).zfill(2) + "/" + str(period).zfill(2) + "_RES")
    track_list = os.listdir(track_result_path)
    track_list = [name for name in track_list if ".tif" in name]
    track_list.sort()
    for file_name in track_list:
        source_file=os.path.join(track_result_path,file_name)
        shutil.copy(source_file,FP_path)

    average_list = get_average_list(img_path, track_path, period)
    record_file=os.path.join(track_result_path,"res_track.txt")
    with open(record_file, "r") as f:
        data = f.readlines()
    lines = [line.strip('\n') for line in data]
    for line in lines:
        line = line.split()
        number = int(line[0])
        start = int(line[1])
        end = int(line[2])
        parent_number = int(line[3])
        if start==(len(track_list)-5):
            break
        #持续帧数小于三帧的情况
        if (end-start)<=2 and parent_number == 0:
            #是否是母细胞
            if (end + 6) > len(track_list):
                continue
            if haveChild(track_path,period,number):
                continue
            elif locationNotInRange(track_result_path,number,end):
                print("{} loaction not in range!".format(number))
                continue
            elif ifPixelValue(img_path,track_path,period,number,start,end,average_list):
                continue
            else:
                #像素值不可判定的情况,利用追踪进行跟踪，在接下来的5帧内附近均未出现新的细胞即为FP
                mask=cv2.imread(os.path.join(track_result_path,track_list[end]),-1)
                mask=(mask==number)*1
                mask_size=np.sum(mask)
                #标记进行膨胀
                for k in range(1, 20, 1):
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
                    mask = mask.astype(np.uint8)
                    mask_new = cv2.dilate(mask, kernel)
                    if (np.sum(mask_new) >= (3 * mask_size)):
                        break
                else:
                    print(number,np.unique(mask_new),np.sum(mask_new),mask_size)
                    isFP=True
                    for m in range(end+1,end+6,1):
                        mask_next=cv2.imread(os.path.join(track_result_path,track_list[m]),-1)
                        overlap=mask_new*mask_next
                        candidates=np.unique(overlap)[1:]
                        for candidate in candidates:
                            start_cand,end_cand,parent_cand=get_information(track_path,period,candidate)
                            if start_cand==m and parent_cand==0:
                                isFP=False
                                break
                    if isFP:
                        print("\t\t\t\tRemoving label {} for track reason".format(number))
                        for j in range(start, end + 1, 1):
                            mask = cv2.imread(os.path.join(FP_path, track_list[j]), -1)
                            FP = (mask == number) * number
                            mask = mask - FP
                            mask = mask.astype(np.uint8)
                            cv2.imwrite(os.path.join(FP_path, track_list[j]), mask)

    for file in track_list:
        mask_without_FP=cv2.imread(os.path.join(FP_path,file),-1)
        mask_without_FP=(mask_without_FP>1)*255
        mask_without_FP=mask_without_FP.astype(np.uint8)
        cv2.imwrite(os.path.join(FP_path,file),mask_without_FP)
track_path="2-Iteration/iteration02/track"
img_path="2-GT/imgs"
period=3
reduceFP_with_track(img_path,"2-Iteration/iteration02/track",2)