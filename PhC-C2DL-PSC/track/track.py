import cv2
import numpy as np
import os



def predict_dataset_2(path, output_path, threshold=0.1):

    # check if path exists
    if not os.path.isdir(path):
        print('input path is not a valid path')
        return

    names = os.listdir(path)
    names = [name for name in names if '.tif' in name]
    names.sort()
    #names = names[:90]
    print("mask for track: {} -- {}".format(names[0],names[-1]))


    img = cv2.imread(os.path.join(path, names[0]), -1)
    mi, ni = img.shape
    print('Relabelling the segmentation masks.')
    records = {}

    old = np.zeros((mi, ni))
    index = 1

    for i, name in enumerate(names):
        result = np.zeros((mi, ni), np.uint16)

        img = cv2.imread(os.path.join(path, name), cv2.IMREAD_ANYDEPTH)

        labels = np.unique(img)[1:]

        parent_cells = []

        for label in labels:

            #对应编号位置为1，背景为0
            mask = (img == label) * 1

            #对应细胞的面积
            mask_size = np.sum(mask)
            '''
            #与前一张图对应位置相乘，找出当前帧该细胞对应位置涵盖上一帧哪几个细胞
            overlap = mask * old

            candidates = np.unique(overlap)[1:]'''

            ######对细胞核进行适当膨胀，得到新的candidates和mask_size
            for j in range(1, 10, 1):
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (j, j))
                mask=mask.astype(np.uint8)
                mask_new = cv2.dilate(mask, kernel)
                if (np.sum(mask_new) >= (1.5 * mask_size)):
                    break
            mask=mask.astype(np.uint16)
            mask_size=np.sum(mask_new)
            overlap=mask_new*old
            candidates=np.unique(overlap)[1:]
            #########

            max_score = 0
            max_candidate = 0

            for candidate in candidates:
                #比较面积
                score = np.sum(overlap == candidate * 1) / mask_size
                #取面积最接近的
                if score > max_score:
                    max_score = score
                    max_candidate = candidate

            if max_score < threshold:
                # no parent cell detected, create new track
                records[index] = [i, i, 0]
                result = result + mask * index
                index += 1
            else:

                if max_candidate not in parent_cells:
                    # prolonging track
                    records[max_candidate][1] = i
                    result = result + mask * max_candidate

                else:
                    # split operations
                    # if have not been done yet, modify original record
                    if records[max_candidate][1] == i:
                        records[max_candidate][1] = i - 1
                        # find mask with max_candidate label in the result and rewrite it to index
                        m_mask = (result == max_candidate) * 1
                        result = result - m_mask * max_candidate + m_mask * index

                        records[index] = [i, i, max_candidate.astype(np.uint16)]
                        index += 1

                    # create new record with parent cell max_candidate
                    records[index] = [i, i, max_candidate.astype(np.uint16)]
                    result = result + mask * index
                    index += 1

                # update of used parent cells
                parent_cells.append(max_candidate)
        # store result
        cv2.imwrite(os.path.join(output_path, "mask"+name), result.astype(np.uint16))
        old = result


    # store tracking
    print('Generating the tracking file.')
    with open(os.path.join(output_path, 'res_track.txt'), "w") as file:
        for key in records.keys():
            file.write('{} {} {} {}\n'.format(key, records[key][0], records[key][1], records[key][2]))

if __name__=="__main__":

    predict_result="data/res_result/"
    track_result="data/track_result/"

    predict_dataset_2(predict_result,track_result)



