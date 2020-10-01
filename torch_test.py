# %%
import torch
from os import path
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import PIL.Image
import pandas as pd
import shutil
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
def train_val_dataset(dataset,val_split = 0.25):
    train_idx,val_idx=train_test_split(list(range(len(dataset))),test_size = 0.25)
    datasets = {}
    datasets['train'] = Subset(dataset,train_idx)
    datasets['test'] = Subset(dataset,val_idx)
    return(datasets)

# %%
def Train(epochs,train_loader,val_loader,criterion,optimizer,device):
    print("===================Starting to train==========================")
    for e in range(epochs):
        train_loss=0
        validation_loss=0
        train_correct=0
        val_correct=0

        net.train()
        for data,labels in train_loader:
            data,labels = data.to(device), labels.to(device)
            optimizer.zeros_grad()
            outputs = net(data)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            _,preds = torch.max(outputs,1)
            train_correct += torch.sum(preds==labels.data)
        
        net.eval()
        for data,labels in val_loader:
            data,labels = data.to(device), labels.to(device)
            val_outputs = net(data)
            val_loss = criterion(val_outputs,labels)
            validation_loss+=val_loss.item()
            _,val_preds = torch.max(val_outputs,1)
            val_correct+=torch.sum(val_preds==labels.data)

        train_loss = train_loss/len(train_dataset)
        train_acc = train_correct.double() / len(train_dataset)
        validation_loss =  validation_loss / len(validation_dataset)
        val_acc = val_correct.double() / len(validation_dataset)
        print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Acuuarcy {:.3f}% \tValidation Acuuarcy {:.3f}%'
                                                           .format(e+1, train_loss,validation_loss,train_acc * 100, val_acc*100))
    torch.save(net.state_dict(),"emo_mode-{}-{}-{}.pt".format(epochs,batchsize,lr))


# %%
curr_data=ImageFolder("F:\\FaceExprDecode\\F001")
print(len(curr_data))

# %%
# Scripts to divide dataset into folders
def concat_csv(csv_file_path, tasks):
    appended_data = []
    for task_i in tasks:
        data = pd.read_csv(csv_file_path+"F001_"+task_i+".csv",header=0)
        data['subject']=task_i
        appended_data.append(data)
    appended_data = pd.concat(appended_data)
    return(appended_data)
f001_csv_path = "F:/FaceExprDecode/F001_label/"
pic_tasks = ["T1","T6","T7","T8"]

f01_csv = concat_csv(f001_csv_path,pic_tasks)

# %%
train_idx,val_idx=train_test_split(list(range(f01_csv.shape[0])),test_size = 0.30, random_state=1)
test_idx,val_idx = train_test_split(val_idx,test_size=0.5,random_state=1)


# %%
def reorganize_pics(csv_file,idx,filepath="F:\\FaceExprDecode\\F001\\",mode="val"):
    """
    Reorganize pictures and create corresponding folders (train/test/valid)
    to contain these pics
    mode can be test, train or val
    """
    _task = np.array(csv_file['subject'])[idx]
    _picnum = np.array(csv_file['1'])[idx]

    _pic_filenames = [filepath+_task[i]+"\\"+str(_picnum[i])+".jpg" for i in range(len(_task))]

    if path.isdir(filepath+mode):
        AA=0
        # TODO: Remove? Requires permission,, complicated

    missing_filenames = []

    for filename in _pic_filenames:
        counter = 0
        while not path.exists(filename):
            counter += 1
            if counter >= 5:
                print("Filename not found for ",filename)
                missing_filenames.append(filename)
                break
            filename = '\\'.join(filename.split("\\")[:-1]+["0"+filename.split("\\")[-1]])
            print(filename)
        if path.exists(filename):
            shutil.copy(filename, filepath+mode)
    return(missing_filenames)


# %%
val_Task = np.array(f01_csv['subject'])[val_idx]
val_picnum = np.array(f01_csv['1'])[val_idx]

# %%
val_pic_filenames = ["F:\\FaceExprDecode\\F001\\"+val_Task[i]+"\\"+str(val_picnum[i])+".jpg" for i in range(len(val_Task))]
# %%
missing_filenames = []
for filename in val_pic_filenames:
    counter = 0
    while not path.exists(filename):
        counter += 1
        if counter >= 5:
            print("FIlename not found for ",filename)
            missing_filenames.append(filename)
            break
        filename = '\\'.join(filename.split("\\")[:-1]+["0"+filename.split("\\")[-1]])
        print(filename)
    if path.exists(filename):
        shutil.copy(filename, "F:\\FaceExprDecode\\F001\\val")

# %%
