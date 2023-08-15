import os.path
from track.track_3_period import *
from tools.tool import *
from concurrent.futures import ProcessPoolExecutor as Pool

def get_average_list(img_path,track_path):
    image_list=os.listdir(img_path)
    image_list.sort()
    track_result_path=os.path.join(track_path,"track_RES")
    track_list=os.listdir(track_result_path)
    track_list = [name for name in track_list if ".tif" in name]
    track_list.sort()

    average_whole_list=[]
    for i,name in enumerate(image_list):
        image=cv2.imread(os.path.join(img_path,name),-1)
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

def get_average(img_path,track_path,label,frame):
    image_list = os.listdir(img_path)
    image_list.sort()
    track_result_path = os.path.join(track_path, "track_RES")
    track_list = os.listdir(track_result_path)
    track_list = [name for name in track_list if ".tif" in name]
    track_list.sort()
    image=cv2.imread(os.path.join(img_path,image_list[frame]),-1)
    mask=cv2.imread(os.path.join(track_result_path,track_list[frame]),-1)
    mask=(mask==label)*1
    area=np.sum(mask)
    value=np.sum(mask*image)
    return value/area

def get_information(track_path,label):
    track_result_path = os.path.join(track_path, "track_RES")
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

def haveChild(track_path,label):
    track_result_path = os.path.join(track_path, "track_RES")
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
    #if x in range(14,754) and y in range(14,722):
    if x in range(14, 1010) and y in range(14, 1010):
        return False
    return True

def ifPixelValue(img_path,track_path,number,start,end,average_list):
    count=0
    for i in range(start, end + 1, 1):
        average = get_average(img_path, track_path, number, i)
        if average < (2 * average_list[i]):
            count += 1
    if count>1:
        return True
    return False

def reduceFP_use_track(img_path, track_path):
    FP_path=os.path.join(track_path,"remove_FP")
    createFolder(FP_path,clean=True)
    image_list = os.listdir(img_path)
    image_list.sort()
    track_result_path = os.path.join(track_path, "track_RES")
    print("------Removing FP in {}------".format(track_result_path))
    track_list = os.listdir(track_result_path)
    track_list = [name for name in track_list if ".tif" in name]
    track_list.sort()
    for file_name in track_list:
        source_file=os.path.join(track_result_path,file_name)
        shutil.copy(source_file,FP_path)

    #开始追踪分析
    record_file=os.path.join(track_result_path,"res_track.txt")
    with open(record_file, "r") as f:
        data = f.readlines()
    lines = [line.strip('\n') for line in data]
    
    checked_candidates=[]
    for line in lines:
        line = line.split()
        number = int(line[0])
        start = int(line[1])
        end = int(line[2])
        parent_number = int(line[3])
        if start==(len(track_list)-5):
            break
        #持续帧数小于三帧的情况
        if (end-start)<=1 and parent_number == 0:
            if start==0:
                continue
            if (end + 9) > len(track_list):
                continue
            # 是否是母细胞
            if haveChild(track_path,number):
                continue
            elif locationNotInRange(track_result_path,number,end):
                continue
            #elif ifPixelValue(img_path,track_path,number,start,end,average_list):
                #continue
            else:
                
                mask=cv2.imread(os.path.join(track_result_path,track_list[end]),-1)
                mask=(mask==number)*1
                mask_size=np.sum(mask)
                #标记进行膨胀
                for k in range(1, 20, 1):
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
                    mask = mask.astype(np.uint8)
                    mask_new = cv2.dilate(mask, kernel)
                    if (np.sum(mask_new) >= (5 * mask_size)):
                        break

                print("\t\t",number,np.unique(mask_new),np.sum(mask_new),mask_size)
                isFP=True
                for m in range(end+1,end+9,1):
                    mask_next=cv2.imread(os.path.join(track_result_path,track_list[m]),-1)
                    overlap=mask_new*mask_next
                    candidates=np.unique(overlap)[1:]
                    if len(candidates)!=0 or number in checked_candidates:
                        isFP=False
                        for cand in candidates:
                            checked_candidates.append(cand)
                        break

                    '''
                    for candidate in candidates:
                        start_cand,end_cand,parent_cand=get_information(track_path,candidate)
                        #if start_cand==m and parent_cand==0:
                        if start_cand == m:
                            isFP=False
                            break
                    '''
                    if not isFP:
                        break

                if isFP:
                    print("\t\t\tRemoving label {} for track reason".format(number))
                    for j in range(start, end + 1, 1):
                        mask = cv2.imread(os.path.join(FP_path, track_list[j]), -1)
                        FP = (mask == number) * number
                        mask = mask - FP
                        mask = mask.astype(np.uint8)
                        cv2.imwrite(os.path.join(FP_path, track_list[j]), mask)

    for file in track_list:
        mask_without_FP=cv2.imread(os.path.join(FP_path,file),-1)
        mask_without_FP=(mask_without_FP>0)*255
        mask_without_FP=mask_without_FP.astype(np.uint8)
        cv2.imwrite(os.path.join(FP_path,file),mask_without_FP)

    print("\tFP has been removed")


def reduceFP_with_Pool(img_path,track_path):
    img_path_list=[]
    track_path_list=[]
    for i in range(3):
        img_path_list.append(img_path)
        track_path_list.append(track_path)
    with Pool() as p:
        p.map(reduceFP_use_track,img_path_list,track_path_list,[1,2,3])

if __name__=="__main__":
    img_path="2-GT/imgs"
    temporal_track_path="2-Iteration/iteration02/temporal/track"
    #reduceFP_use_track(img_path,track_path,1)
    #reduceFP_with_Pool(img_path,track_path)
    reduceFP_use_track(img_path, temporal_track_path)
