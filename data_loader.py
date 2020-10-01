import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

task_series=['T1','T6','T7','T8']
task_labels=['Happiness','Embarassment','Fear or nervous','Pain']

# https://github.com/omarsayed7/Deep-Emotion/blob/master/data_loaders.py
class image_Loader(Dataset):
    def __init__(self,csv_file,img_dir,transform, action_unit):

        #==============Labels concatenates============
        self.csv_file = csv_file
        self.action_unit = action_unit
        self.labels = self.csv_file[str(self.action_unit)]
        #==============Images===========================
        #self.img_subset = self.csv_file['0']
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return(len(self.csv_file))
        
    def __getitem__(self,idx):
        #print("step 0,",idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = Image.open(self.img_dir+str(self.img_subset[idx])+".jpg")
        #print("hello",self.labels[idx])
        #print(type(self.labels[idx]))
        labels = np.array(self.labels[idx])
        labels = torch.from_numpy(labels).long()
        #print("label:", labels)
        if self.transform:
        #    print("transform")
            img0 = self.transform(img)
        #    print("transform complete")
        #print("almost finihsed")
        return(img0,labels)
    
def eval_data_dataloader(csv_file,img_dir,sample_number,transform=None):
    if transform is None:
        transform = transforms.Compose([
                    transforms.Resize(257),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = image_Loader(csv_file=csv_file,img_dir=img_dir,transform=transform)
    #print("Transform:",transform)
    label = dataset.__getitem__(sample_number)[1]
    #print(label)
    imgg = dataset.__getitem__(sample_number)[0]
    imgt = (imgg.permute(1,2,0)).numpy()
    #print(imgnp.shape)
    #imgt = imgnp.squeeze()
    plt.imshow(imgt)
    plt.show()

#eval_data_dataloader(csv_file="F:\FaceExprDecode\demo_labels\F001_T1.csv", img_dir="F:/FaceExprDecode/demo_pics/",sample_number=10)

