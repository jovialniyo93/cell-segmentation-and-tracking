import cv2
import os
import numpy as np
import time
from tqdm import tqdm
def get_center(serial,label,directory):
    track_picture = os.listdir(directory)
    track_picture = [file for file in track_picture if ".tif" in file]
    track_picture.sort()
    result_picture = cv2.imread(os.path.join(directory, track_picture[serial]), -1)
    label_picture = ((result_picture == label) * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(label_picture, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00']==0 :
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx,cy)

def get_trace(image_path,track_path,trace_path,text=False):
    track_picture=os.listdir(track_path)
    track_picture = [file for file in track_picture if ".tif" in file]
    track_picture.sort()
    print("\ttrack result: {} -- {}".format(track_picture[0],track_picture[-1]))
    length_track=len(track_picture)

    test_image = os.listdir(image_path)
    test_image.sort()
    trace_image=[]
    print("\toriginal images: {} -- {}".format(test_image[0],test_image[-1]))
    for i in range(len(test_image)):
        image_to_draw = cv2.imread(os.path.join(image_path, test_image[i]), -1)
        image_to_draw = np.stack((image_to_draw,) * 3, axis=2)
        result_picture = cv2.imread(os.path.join(track_path, track_picture[i]), -1)
        label_picture = ((result_picture >= 1) * 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(label_picture, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_to_draw = cv2.drawContours(image_to_draw, contours, -1, (0, 0, 230), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if text:
            for contour in contours:
                M = cv2.moments(contour)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(image_to_draw, str(result_picture[cy][cx]), (cx, cy), font, 0.5, (255, 255, 255), 1)
        trace_image.append(image_to_draw)


    file = os.path.join(track_path,"res_track.txt")
    with open(file, "r") as f:
        data = f.readlines()
    lines = [line.strip('\n') for line in data]
    for line in lines:
        line = line.split()
        number = int(line[0])
        start = int(line[1])
        end = int(line[2])
        parent_number = int(line[3])

        if start != end and parent_number == 0:
            center = get_center(start, number,track_path)
            if center != None:
                cv2.circle(trace_image[start], center, 3, (0, 255, 0), -1)
            start_point = center
            for i in range(start + 1, end + 1):
                center = get_center(i, number,track_path)
                if center == None:
                    continue
                else:

                    cv2.circle(trace_image[i], center, 3, (255, 0, 0), -1)
                    for j in range(start, i):
                        cv2.line(trace_image[j], start_point, center, (255, 255, 255))
                    start_point = center

        if start != end and parent_number != 0:
            parent_point = get_center(start - 1, parent_number,track_path)
            for i in range(start, end + 1):
                center = get_center(i, number,track_path)
                if center == None:
                    continue
                else:
                    cv2.circle(trace_image[i], center, 3, (255, 0, 0), -1)
                    for j in range(start - 1, i):
                        cv2.line(trace_image[j], parent_point, center, (255, 255, 255))
                    parent_point = center

        end_center = get_center(end, number, track_path)
        if end_center != None:
            cv2.circle(trace_image[end], end_center, 3, (0, 0, 255), -1)

    for i in range(len(trace_image)):
        original=cv2.imread(os.path.join(image_path,test_image[i]),-1)
        original=cv2.cvtColor(original,cv2.COLOR_GRAY2RGB)
        cat_img=np.hstack((original,trace_image[i]))
        cv2.imwrite(os.path.join(trace_path, str(i).zfill(6)+".tif"), cat_img)
    print("\ttrace: {} -- {} has been generated.".format(test_image[0],test_image[-1]))


def get_video(trace_path):
    directory = trace_path
    pictures = os.listdir(directory)
    picture_names = [name for name in pictures if "trace" not in name]
    picture_names.sort()
    print("\t\tGenerating video.")
    print("\t\ttrace image:{} -- {}".format(picture_names[0],picture_names[-1]))
    picture_names.sort()
    fps = 1  # 视频每秒1帧 cv2.VideoWriter_fourcc(*'XVID')
    image = cv2.imread(os.path.join(directory, picture_names[0]), -1)
    size = (image.shape[1], image.shape[0])  # 需要转为视频的图片的尺寸,  可以使用cv2.resize()进行修改

    #videowriter = cv2.VideoWriter(os.path.join(trace_path,"trace.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    videowriter = cv2.VideoWriter(os.path.join(trace_path, "trace.avi"), cv2.VideoWriter_fourcc(*"XVID"), fps, size)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_array=[]
    for i, name in enumerate(picture_names):
        img = cv2.imread(os.path.join(directory, name), -1)
        cv2.putText(img, str(i), (780, 30), font, 1, (255,255,255), 3)
        img_array.append(img)
    for img in img_array:
        videowriter.write(img)
    print("\t\tVideo for trace image {} to {} has been generated.".format(0,len(picture_names)))




