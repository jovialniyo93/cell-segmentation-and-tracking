import os.path
from track import *
from generate_trace import *
from multiprocessing import Pool
from tools.tool import *


def get_unusual(track_result_path,unusual_result_path,label_num):
    createFolder(unusual_result_path)
    file = os.path.join(track_result_path, "res_track.txt")
    with open(file, "r") as f:
        data = f.readlines()
    lines = [line.strip('\n') for line in data]
    for line in lines:
        line = line.split()
        number = int(line[0])
        if number==label_num:
            start = int(line[1])
            end = int(line[2])
            parent_number = int(line[3])
            break
    track_result=os.listdir(track_result_path)
    track_result=[name for name in track_result if ".tif" in name]
    track_result.sot()
    print("Get unusual cell {} for image {} to {}:".format(label_num,start,end))
    for i in range(start,end+1,1):
        mask=cv2.imread(os.path.join(track_result_path,track_result[i]),-1)
        unusual_result = (mask == label_num) * 1
        unusual_result=unusual_result.astype(np.uint8)
        result_path=os.path.join(unusual_result_path,track_result[i].replace("mask",""))
        if os.path.isfile(result_path):
            new_mask=cv2.imread(result_path,-1)
            unusual_result+=new_mask
        cv2.imwrite(result_path,unusual_result)

def remove_unusual(unusual_path,source_path,result_path):
    copyFile(source_path,result_path)
    unusual_list=os.listdir(unusual_path)
    unusual_list.sort()
    for mask_name in unusual_list:
        print("Removing unusual cell in {}:",mask_name)
        source_img=cv2.imread(os.path.join(source_path,mask_name),-1)
        unusual_cell=cv2.imread(os.path.join(unusual_path,mask_name),-1)
        source_img-=unusual_cell
    cv2.imwrite(os.path.join(result_path,mask_name),source_img)



if __name__=="__main__":

    #track_all_periods("7/imgs","7",remove_edge=False,mask_text=True,prefix_name="")
    a=np.array([1,0,1,0,0])
    print(-a)