from track.track import *
from track.generate_trace import *

from concurrent.futures import ProcessPoolExecutor as Pool
from tools.tool import *

def process_predictResult(source_path,result_path,period=0,remove_edge=False):
    names = os.listdir(source_path)
    names = [name for name in names if '.tif' in name]
    if len(names)!=233:
        names=[names[i] for i in range(len(names)) if i%8==0]
    names.sort()
    if period==1:
        names=names[:102]
    if period==2:
        names=names[102:143]
    if period==3:
        names=names[143:]
    for i,name in enumerate(names):
        predict_result=cv2.imread(os.path.join(source_path,name),-1)
        height = predict_result.shape[0]
        if height!=736:
            predict_result=predict_result[5: 741, 1: 769]
        ret, predict_result = cv2.threshold(predict_result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        predict_result = useAreaFilter(predict_result, 40)
        #连通域检测、编号
        if remove_edge:
            contours, hierarchy = cv2.findContours(predict_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.stack((predict_result,) * 3, axis=2)
            for j, cont in enumerate(contours):
                for coordinate in cont:
                    x=coordinate[0][0]
                    y=coordinate[0][1]
                    if x==0 or x==767 or y==0 or y==735:
                        #print(coordinate)
                        cv2.drawContours(mask, contours, j, (0, 0, 0), thickness=-1)
                        break
            mask = mask[:, :, 0]
            mask = (mask != 0) * 255
            mask = mask.astype(np.uint8)
            predict_result=mask

        ret, markers = cv2.connectedComponents(predict_result,ltype=2)
        cv2.imwrite(os.path.join(result_path,str(i).zfill(6)+".tif"),markers)

def track_single_period(img_path,track_path,mask_path,period):
    period_path=os.path.join(track_path,str(period).zfill(2))
    createFolder(period_path)
    predict_path = os.path.join(period_path, "predict_result")
    track_result_path = os.path.join(period_path, str(period).zfill(2)+"_RES")
    trace_path = os.path.join(period_path, "trace")
    createFolder(predict_path,clean=True)
    createFolder(track_result_path)
    createFolder(trace_path)

    process_predictResult(mask_path,predict_path,period,remove_edge=True)#二值化+过滤+去边界细胞+连通域编号
    predict_dataset_2(predict_path,track_result_path)#记得改输出的名字  GT：man_track***  RES:res_track
    get_trace(img_path, track_result_path, trace_path,period,text=True)#记得加上原图裁剪
    get_video(trace_path)

def track_all_periods(img_path,mask_dir,track_path,remove_edge=True):
    track_dir=track_path
    createFolder(track_dir)
    img_path_list=[]
    track_dir_list=[]
    mask_dir_list=[]
    for i in range(1, 4):
        img_path_list.append(img_path)
        track_dir_list.append(track_dir)
        mask_dir_list.append(mask_dir)

    #track_single_period(img_path,track_path,mask_dir,2)
    with Pool() as p:
        p.map(track_single_period,img_path_list,track_dir_list,mask_dir_list,range(1,4))


if __name__=="__main__":
    '''folder="data/"
    test_path = os.path.join(folder, "imgs")
    mask_path=os.path.join(folder,"mask")
    predict_path=os.path.join(folder,"predict_result")
    track_result_path = os.path.join(folder, "track_result")
    trace_path = os.path.join(folder, "trace")

    process_predictResult(mask_path,predict_path)
    predict_dataset_2(predict_path,track_result_path)

    get_trace(test_path, track_result_path, trace_path)
    get_video(trace_path)'''


    mask_list=os.listdir("data/mask_before_process/")
    mask_list.sort()
    for name in mask_list:
        mask=cv2.imread(os.path.join("data/mask_before_process",name),-1)
        ret, mask_new = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask_new = useAreaFilter(mask_new, 40)
        cv2.imwrite(os.path.join("data/mask",name),mask_new)
    track_all_periods("data/imgs","data/mask","data/track")

