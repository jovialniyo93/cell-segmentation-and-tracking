from detect.test import *
from multiprocessing import Pool
from detect.train import *
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
backbone = 'resnet50'
pretrained = 'imagenet'
DEVICE = 'cuda'

def augBC(imgs_dir,imgs_aug_dir,sequence,mask_dir,mask_aug_dir,mask):
    contrast_list = [0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]
    contrast_type=len(contrast_list)+1
    img_num=sequence
    img_name=str(sequence).zfill(6)+".tif"
    img=cv2.imread(os.path.join(imgs_dir,img_name),-1)
    cv2.imwrite(os.path.join(imgs_aug_dir,str(img_num*contrast_type).zfill(6)+".tif"),img)
    if mask:
        img_mask=cv2.imread(os.path.join(mask_dir,img_name),-1)
        for i in range(contrast_type):
            cv2.imwrite(os.path.join(mask_aug_dir, str(img_num * contrast_type+i).zfill(6) + ".tif"), img_mask)
    img=img.astype(np.float32)
    print("generating contrast augmentation for image:",img_name)
    for j,contrast in enumerate(contrast_list,start=1):
        img_C=np.clip(img*contrast,0,255)
        img_C=img_C.astype(np.uint8)
        cv2.imwrite(os.path.join(imgs_aug_dir,str(img_num*contrast_type+j).zfill(6)+".tif"),img_C)

def augmentationWithPool(imgs_dir,imgs_aug_dir,mask_dir=None,mask_aug_dir=None,mask=False):
    createFolder(imgs_aug_dir,clean=True)
    if mask:
        createFolder(mask_aug_dir,clean=True)
    img_list=os.listdir(imgs_dir)
    img_list.sort()
    print(img_list)
    p=Pool()
    for i in range(len(img_list)):
        p.apply_async(augBC,args=(imgs_dir,imgs_aug_dir,i,mask_dir,mask_aug_dir,mask,))
    p.close()
    p.join()

def test_and_process(img_path,result_path,process_result_path,mask_new_before_process_path,mask_new_path,ckpt_path,mask_old_path):
    createFolder(result_path,clean=True)
    createFolder(process_result_path,clean=True)
    createFolder(mask_new_path,clean=True)
    createFolder(mask_new_before_process_path, clean=True)
    ckpt_list=os.listdir(ckpt_path)
    ckpt_list.sort()
    final_ckpt=os.path.join(ckpt_path,ckpt_list[-1])
    print("predicting images with ckpt:",final_ckpt)
    test(img_path,result_path,final_ckpt)
    result_list=os.listdir(result_path)
    result_list=[name for name in result_list if ".tif" in name]
    result_list.sort()
    result_imgs=[]
    print("\tprocessing predicted mask:",result_path)
    for result_name in result_list:
        result=cv2.imread(os.path.join(result_path,result_name),-1)
        img=result.astype(np.float32)
        result_imgs.append(img)
        ret,result=cv2.threshold(result,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #result=useAreaFilter(result,40)
        cv2.imwrite(os.path.join(process_result_path,result_name),result)
    print("\tgenerating new mask:",mask_new_path)

    contrast_type=int(len(result_imgs)/233)
    for i in range(0,len(result_imgs),contrast_type):
        mask_combine=np.zeros((result_imgs[i].shape),dtype=np.float32)
        for j in range(i,i+contrast_type):
            mask_combine+=result_imgs[j]*0.125
        mask_new=mask_combine.astype(np.uint8)
        cv2.imwrite(os.path.join(mask_new_before_process_path, str(i//contrast_type).zfill(6) + ".tif"), mask_new)
        ret, mask_new = cv2.threshold(mask_new, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask_new = useAreaFilter(mask_new, 40)
        for k in range(contrast_type):
            mask_old=cv2.imread(os.path.join(mask_old_path,str(i+k).zfill(6)+".tif"),-1)
            if mask_old.shape[0]!=736:
                mask_old=mask_old[5:741,1:769]
            mask_old=mask_old.astype(np.float32)
            mask_final=0.5*mask_old+0.5*mask_combine
            mask_final=mask_final.astype(np.uint8)
            ret, mask_final = cv2.threshold(mask_final, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask_final = useAreaFilter(mask_final, 40)
            cv2.imwrite(os.path.join(mask_new_path,str(i+k).zfill(6)+".tif"),mask_final)

def trainWithIteration(img_path,iteration_path,keep_training,iteration_times=5,batch_size=32,data_shuffle=False):
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    y_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    model = smp.DeepLabV3Plus(backbone, in_channels=1, encoder_weights=pretrained)
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters())
    ckpt_path = os.path.join(iteration_path, "checkpoints")
    createFolder(ckpt_path,clean=True)

    for i in range(iteration_times):
        if i==0:
            training=False
        else:
            training=keep_training

        current_folder=os.path.join(iteration_path,"iteration"+str(i+1).zfill(2))
        mask_path=os.path.join(current_folder,"mask")
        print("\nstart iteration {}/{}: {}-{}".format(i + 1,iteration_times, img_path,mask_path))
        data = TrainDataset(img_path, mask_path, x_transforms, y_transforms)
        dataloader = DataLoader(data, batch_size, shuffle=data_shuffle, num_workers=4)

        train_model(model, criterion, optimizer, dataloader, training, ckpt_path, num_epochs=1)
        result_path=os.path.join(current_folder,"predict")
        process_result_path=os.path.join(current_folder,"predict_result")
        mask_new_before_process_path=os.path.join(current_folder,"weighted_sum")
        next_folder=os.path.join(iteration_path,"iteration"+str(i+2).zfill(2))
        createFolder(next_folder)
        mask_new_path=os.path.join(next_folder,"mask")
        test_and_process(img_path,result_path,process_result_path,mask_new_before_process_path,mask_new_path,ckpt_path,mask_path)


if __name__=="__main__":
    img_path = "data/imgs/"
    mask_path = "data/mask/"
    img_aug_path = "data/augmentation/imgs/"
    mask_aug_path = "data/augmentation/mask/"
    data_path = "data/augmentation/"
    ckpt_path = "checkpoints/"
    '''
    img_path="data/imgs/"
    mask_path="data/mask/"
    img_aug_path="iteration/imgs_aug/"
    mask_aug_path="iteration/iteration1/mask"
    iteration_path = "iteration"
    '''
    augmentationWithPool(img_path, img_aug_path, mask_path, mask_aug_path, mask=True)

    trainWithIteration(data_path, ckpt_path, iteration_times=10)
