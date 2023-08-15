from torch.utils.data import Dataset
import cv2
import os
import numpy as np
import torch
from torchvision.transforms import transforms

def train_dataset(img_root,mask_root):
    imgs=[]
    n=len(os.listdir(img_root))
    for i in range(n):
        img=os.path.join(img_root,str(i).zfill(6)+".tif")
        mask=os.path.join(mask_root,str(i).zfill(6)+".tif")
        imgs.append((img,mask))
    return imgs

def test_dataset(img_root):
    imgs=[]
    n=len(os.listdir(img_root))

    for i in range(n):
        img = os.path.join(img_root, str(i).zfill(6)+ ".tif")
        imgs.append(img)
    return imgs


class TrainDataset(Dataset):
    def __init__(self,img_root,mask_root,transform=None,mask_transform=None):
        imgs=train_dataset(img_root,mask_root)
        self.imgs=imgs
        self.transform=transform
        self.mask_transform=mask_transform

    def __getitem__(self, index):
        x_path,y_path=self.imgs[index]
        img_x=cv2.imread(x_path,-1)
        img_y=cv2.imread(y_path,-1)
        if self.transform is not None:
            img_x=self.transform(img_x)
        if self.mask_transform is not None:
            img_y=self.mask_transform(img_y)

        return img_x,img_y

    def __len__(self):
        return len(self.imgs)

class TestDataset(Dataset):
    def __init__(self, img_root, transform=None):
        imgs = test_dataset(img_root)
        self.imgs = imgs
        self.transform=transform

    def __getitem__(self, index):
        x_path = self.imgs[index]
        img_x = cv2.imread(x_path,-1)
        if self.transform is not None:
            img_x=self.transform(img_x)
        return img_x

    def __len__(self):
        return len(self.imgs)
