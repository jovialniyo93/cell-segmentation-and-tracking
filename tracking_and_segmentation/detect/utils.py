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
        #img=os.path.join(img_root,"t"+str(i).zfill(3)+".tif")
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
        img_x=img_x[5:741,1:769]
        if img_y.shape[0]!=736:
            img_y=img_y[5:741,1:769]
        if self.transform is not None:
            #print(img_x.dtype)
            img_x=self.transform(img_x)
        if self.mask_transform is not None:
            '''
            img_y = img_y / 1.0
            for i in range(img_y.shape[0]):
                for j in range(img_y.shape[1]):
                    img_y[i][j] = 0 if img_y[i][j]==0 else 1
            img_y = img_y.astype(np.float16)
            '''
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
        img_x = img_x[5:741,1:769]
        if self.transform is not None:
            img_x=self.transform(img_x)
        return img_x

    def __len__(self):
        return len(self.imgs)

if __name__=="__main__":
    model_path = 'checkpoints/'
    imgs_path = 'data/imgs/'
    mask_path = 'data/mask/'
    # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # cv2.equalizeHist(image) / 255

    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.5], [0.5])
    ])

    y_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    data = TrainDataset(imgs_path, mask_path, x_transforms,y_transforms)
    img,mask=data[2]
    print(np.unique(img.numpy()),img.shape,img.dtype)
    print(np.unique(mask.numpy()),mask.shape,mask.dtype)
    test_data = TestDataset("data/test", transform=x_transforms)
    print(np.unique(test_data[0]),test_data[0].dtype)
