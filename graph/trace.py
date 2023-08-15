import cv2
import os
import numpy as np
import time
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def get_center(serial, label, directory):
    track_picture = os.listdir(directory)
    track_picture = [file for file in track_picture if ".tif" in file]
    track_picture.sort()
    result_picture = cv2.imread(os.path.join(directory, track_picture[serial]), -1)
    label_picture = ((result_picture == label) * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(label_picture, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)


def get_coloured_mask(mask):
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [255, 128, 0],
               [128, 255, 0], [0, 255, 128], [0, 128, 255], [128, 0, 255], [255, 0, 128], [80, 70, 180], [250, 80, 190],
               [245, 145, 50], [70, 150, 250], [50, 190, 190], [0, 128, 0], [255, 165, 0]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def get_trace(image_path, track_path, trace_path, period, text=False):
    track_picture = os.listdir(track_path)
    track_picture = [file for file in track_picture if ".tif" in file]
    track_picture.sort()
    print("\ttrack result: {} -- {}".format(track_picture[0], track_picture[-1]))
    length_track = len(track_picture)

    test_image = os.listdir(image_path)
    test_image.sort()
    if period == 1:
        test_image = test_image[:102]
    if period == 2:
        test_image = test_image[102:143]
    if period == 3:
        test_image = test_image[143:]
    trace_image = []
    print("\toriginal images: {} -- {}".format(test_image[0], test_image[-1]))
    for i in range(len(test_image)):
        image_to_draw = cv2.imread(os.path.join(image_path, test_image[i]), -1)
        image_to_draw = image_to_draw[1:830, 1:990]
        image_to_draw = np.stack((image_to_draw,) * 3, axis=2)
        result_picture = cv2.imread(os.path.join(track_path, track_picture[i]), -1)
        label_picture = ((result_picture >= 1) * 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(label_picture, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_to_draw = cv2.drawContours(image_to_draw, contours, -1, (0, 0, 255), 2)
        if text:
            for contour in contours:
                M = cv2.moments(contour)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(image_to_draw, str(result_picture[cy][cx]), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Apply the colored mask to the image
        for contour in contours:
            mask = np.zeros_like(image_to_draw[:, :, 0])
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            colored_mask = get_coloured_mask(mask)
            image_to_draw = cv2.addWeighted(image_to_draw, 1, colored_mask, 0.5, 0)

        trace_image.append(image_to_draw)

    file = os.path.join(track_path, "res_track.txt")
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
            center = get_center(start, number, track_path)
            if center is not None:
                cv2.circle(trace_image[start], center, 3, (0, 255, 0), -1)
            start_point = center
            for i in range(start + 1, end + 1):
                center = get_center(i, number, track_path)
                if center is None:
                    continue
                else:
                    cv2.circle(trace_image[i], center, 3, (0, 255, 0), -1)  # Green circle for the trajectory
                    for j in range(start, i):
                        cv2.line(trace_image[j], start_point, center, (0, 255, 0), 2)  # Green line for the trajectory
                    start_point = center

        if start != end and parent_number != 0:
            parent_point = get_center(start - 1, parent_number, track_path)
            for i in range(start, end + 1):
                center = get_center(i, number, track_path)
                if center is None:
                    continue
                else:
                    cv2.circle(trace_image[i], center, 3, (0, 255, 0), -1)  # Green circle for the trajectory
                    for j in range(start - 1, i):
                        cv2.line(trace_image[j], parent_point, center, (0, 255, 0), 2)  # Green line for the trajectory
                    parent_point = center

        end_center = get_center(end, number, track_path)
        if end_center is not None:
            cv2.circle(trace_image[end], end_center, 3, (0, 0, 255), -1)

    for i in range(len(trace_image)):
        original = cv2.imread(os.path.join(image_path, test_image[i]), -1)
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
        original = original[1:830, 1:990, :]
        cat_img = np.hstack((original, trace_image[i]))
        cv2.imwrite(os.path.join(trace_path, str(i).zfill(6) + ".tif"), cat_img)
    print("\ttrace: {} -- {} has been generated.".format(test_image[0], test_image[-1]))


def get_video(trace_path):
    directory = trace_path
    pictures = os.listdir(directory)
    picture_names = [name for name in pictures if "trace" not in name]
    print("\t\tGenerating video.")
    print("\t\ttrace image:{} -- {}".format(picture_names[0], picture_names[-1]))
    picture_names.sort()
    fps = 1
    image = cv2.imread(os.path.join(directory, picture_names[0]), -1)
    size = (image.shape[1], image.shape[0])

    videowriter = cv2.VideoWriter(os.path.join(trace_path, "trace.avi"), cv2.VideoWriter_fourcc(*"XVID"), fps, size)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_array = []
    for i, name in enumerate(picture_names):
        img = cv2.imread(os.path.join(directory, name), -1)
        cv2.putText(img, str(i), (780, 30), font, 1.5, (0, 0, 255), 3)
        img_array.append(img)
    for img in img_array:
        videowriter.write(img)
    print("\t\tVideo for trace image {} to {} has been generated.".format(0, len(picture_names)))


image_path = "8bit_imgs/"
track_path = "track_result/"
trace_path = "trace_result/"
period = 0

get_trace(image_path, track_path, trace_path, period, text=True)
get_video(trace_path)
