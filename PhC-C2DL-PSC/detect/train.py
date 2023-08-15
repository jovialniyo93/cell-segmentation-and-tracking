import torch
from torch.utils.data import DataLoader
from torch import nn,optim
from detect.utils import *
import numpy as np
import cv2
from tqdm import tqdm
import logging
from torchvision.transforms import transforms
import os
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
backbone = 'resnet50'
pretrained = 'imagenet'
DEVICE = 'cuda'



def __normalize(mask):
    min,max=np.unique(mask)[0],np.unique(mask)[-1]
    mask=mask/1.0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask[i][j]=(mask[i][j]-min)/(max-min)
    mask = mask.astype(np.float16)
    return mask

def record_result(string):
    file_name="train_record.txt"
    if not os.path.exists(file_name):
        with open(file_name,'w') as f:
            print("successfully create record file")
    with open(file_name,'a') as f:
        f.write(string+"\n")
    print(string+" has been recorded")

def train_model(model,criterion,optimizer,dataload,keep_training,ckpt_path,num_epochs=50):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count()>1:
        model=nn.DataParallel(model)
    model.to(device)

    if keep_training:
        checkpoints=os.listdir(ckpt_path)
        checkpoints.sort()
        final_ckpt=checkpoints[-1]
        print("Continue training from ",final_ckpt)
        restart_epoch=final_ckpt.replace("CP_epoch","").replace(".pth","")
        restart_epoch=int(restart_epoch)
        model.load_state_dict(torch.load(os.path.join(ckpt_path,final_ckpt)))

    else:
        restart_epoch=0
        if os.path.isfile("train_record.txt"):
            os.remove("train_record.txt")
            print("Old results' record has been cleaned!")

    for epoch in range(num_epochs):
        model.train()
        print('Epoch {}'.format(restart_epoch+epoch+1))
        data_size=len(dataload.dataset)
        epoch_loss=0
        step=0
        for x,y in tqdm(dataload):
            step+=1
            inputs=x.to(device)
            labels=y.to(device)
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
        print("epoch %d loss:%.3f"%(restart_epoch+epoch+1,epoch_loss/step))
        record_result("epoch %d loss:%.3f"%(restart_epoch+epoch+1,epoch_loss/step))
        try:
            os.mkdir(ckpt_path)
            logging.info('Created checkpoint directory')
        except OSError:
            pass
        torch.save(model.state_dict(),os.path.join(ckpt_path , f'CP_epoch{str(restart_epoch+epoch+1).zfill(2)}.pth'))
        logging.info(f'Checkpoint {restart_epoch+epoch + 1} saved !')
        return epoch_loss

if __name__=="__main__":
    ckpt_path = 'checkpoints/'
    imgs_path = 'data/imgs/'
    mask_path = 'data/mask/'
    # cv2.equalizeHist(image) / 255

    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    y_transforms = transforms.Compose([
        # transforms.Lambda(lambda mask:__normalize(mask)),
        transforms.ToTensor()
    ])

    keep_training=False
    model = smp.DeepLabV3Plus(backbone, in_channels=1, encoder_weights=pretrained)
    batch_size=8
    criterion=nn.BCEWithLogitsLoss()
    optimizer=optim.Adam(model.parameters())
    data=TrainDataset(imgs_path,mask_path,x_transforms,y_transforms)
    dataloader=DataLoader(data,batch_size,shuffle=True,num_workers=4)
    train_model(model,criterion,optimizer,dataloader,keep_training,ckpt_path,num_epochs=10)



