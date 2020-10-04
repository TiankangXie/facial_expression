import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import glob
from skimage import io

task_series=['T1','T6','T7','T8']
task_labels=['Happiness','Embarassment','Fear or nervous','Pain']

# https://github.com/omarsayed7/Deep-Emotion/blob/master/data_loaders.py
class image_Loader(Dataset):
    def __init__(self,csv_dir,img_dir,transform,action_unit):

        #==============Labels concatenates============
        self.csv_file = pd.read_csv(csv_dir)
        self.action_unit = action_unit
        #self.labels = 
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

        #"F:/Summer2019/FaceExprDecode/F001/T1/**.jpg"

        img_filename = self.img_dir+self.csv_file['task'][idx]+"\\"+str(self.csv_file['1'][idx])+".jpg"
        counter=0
        while not os.path.exists(img_filename):
            #print(img_filename)
            counter += 1
            if counter >= 5:
                print("Filename not found for ",img_filename)
                raise FileNotFoundError("We failed to find this file")
                break
            img_filename = '\\'.join(img_filename.split("\\")[:-1]+["0"+img_filename.split("\\")[-1]])

        img = Image.open(img_filename)
        #img = io.imread(img_filename)

        y_labels = torch.tensor(int(self.csv_file[str(self.action_unit)][idx]))

        if self.transform:
            img = self.transform(img)

        return(img,y_labels)
    
def eval_data_dataloader(csv_file,img_dir,sample_number,action_unit,transform=None):
    if transform is None:
        transform = transforms.Compose([
                    transforms.Resize(257),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = image_Loader(csv_dir=csv_file,img_dir=img_dir,transform=transform,action_unit="32")
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
#eval_data_dataloader(train_csv,img_dir="F:\\FaceExprDecode\\F001\\train\\",sample_number=5,action_unit="32")
# eval_data_dataloader(csv_file="F:\\FaceExprDecode\\F001\\train\\train.csv",img_dir="F:\\FaceExprDecode\\F001\\",sample_number=10,action_unit="32")