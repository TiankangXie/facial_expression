# %%
import sys, importlib

import torch
import os
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms,models
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import PIL.Image
import pandas as pd
import shutil
from data_loader import image_Loader
import torch.optim as optim
from emo_model import AU_model

#importlib.reload(sys.modules['emo_model'])
#from emo_model import AU_model

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
            #net.cleargrads()
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
        data['task']=task_i
        appended_data.append(data)
    appended_data = pd.concat(appended_data)
    return(appended_data)
f001_csv_path = "F:/FaceExprDecode/F001_label/"
pic_tasks = ["T1","T6","T7","T8"]

f01_csv = concat_csv(f001_csv_path,pic_tasks).reset_index()

# %%
train_idx,val_idx=train_test_split(list(range(f01_csv.shape[0])),test_size = 0.30, random_state=1)
test_idx,val_idx = train_test_split(val_idx,test_size=0.5,random_state=1)


# %%
def reorganize_pics(csv_file,idx,filepath="F:\\FaceExprDecode\\F001\\",mode="val\\pictures"):
    """
    Reorganize pictures and create corresponding folders (train/test/valid)
    to contain these pics
    mode can be test, train or val
    Example usage: 
    """
    _task = np.array(csv_file['subject'])[idx]
    _picnum = np.array(csv_file['1'])[idx]

    _pic_filenames = [filepath+_task[i]+"\\"+str(_picnum[i])+".jpg" for i in range(len(_task))]

    if os.path.isdir(filepath+mode):
        print("True")
        shutil.rmtree(filepath+mode)
    os.mkdir(filepath+mode)
        #os.chmod(filepath+mode,stat.S_IRWXU)
        #AA=0
        # TODO: Remove? Requires permission,, complicated

    missing_filenames = []

    for filename in _pic_filenames:
        counter = 0
        while not os.path.exists(filename):
            counter += 1
            if counter >= 5:
                print("Filename not found for ",filename)
                missing_filenames.append(filename)
                break
            filename = '\\'.join(filename.split("\\")[:-1]+["0"+filename.split("\\")[-1]])
            print(filename)
        if os.path.exists(filename):
            shutil.copy(filename, filepath+mode)
    return(missing_filenames)


# %%
val_Task = np.array(f01_csv['subject'])[val_idx]
val_picnum = np.array(f01_csv['1'])[val_idx]

# %%
val_pic_filenames = ["F:\\FaceExprDecode\\F001\\"+val_Task[i]+"\\"+str(val_picnum[i])+".jpg" for i in range(len(val_Task))]
# %%
# missing_filenames = []
# for filename in val_pic_filenames:
#     counter = 0
#     while not os.path.exists(filename):
#         counter += 1
#         if counter >= 5:
#             print("FIlename not found for ",filename)
#             missing_filenames.append(filename)
#             break
#         filename = '\\'.join(filename.split("\\")[:-1]+["0"+filename.split("\\")[-1]])
#         print(filename)
#     if os.path.exists(filename):
#         shutil.copy(filename, "F:\\FaceExprDecode\\F001\\val")
# %%
f01_csv.iloc[train_idx,:].to_csv("F:\\FaceExprDecode\\F001\\train\\train.csv")
f01_csv.iloc[test_idx,:].to_csv("F:\\FaceExprDecode\\F001\\test\\test.csv")
f01_csv.iloc[val_idx,:].to_csv("F:\\FaceExprDecode\\F001\\val\\val.csv")

# %%
train_dir = "F:\\FaceExprDecode\\F001\\train\\"
test_dir = "F:\\FaceExprDecode\\F001\\test\\"
val_dir = "F:\\FaceExprDecode\\F001\\val\\"

train_csv = train_dir+"train.csv"
test_csv = test_dir+"test.csv"
val_csv = val_dir+"val.csv"

train_dataset = image_Loader(csv_dir=train_csv,img_dir=train_dir+"pictures\\",transform=None,action_unit=10)
test_dataset = image_Loader(csv_dir=test_csv,img_dir=test_dir+"pictures\\",transform=None,action_unit=10)
val_dataset = image_Loader(csv_dir=val_csv,img_dir=val_dir+"pictures\\",transform=None,action_unit=10)

train_loader = DataLoader(train_dataset,batch_size=50,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size = 50, shuffle=True)

# %%
net = AU_model()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.001)
# %%
transform = transforms.Compose([
            transforms.Resize(257),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
Dataset01 = image_Loader(csv_dir="F:\\here.csv",img_dir="F:\\FaceExprDecode\\F001\\",transform=transform, action_unit='32')
train_set,test_set = torch.utils.data.random_split(Dataset01,[1000,230])
train_loader = DataLoader(dataset=train_set,batch_size=50,shuffle=True)
test_loader = DataLoader(dataset=test_set,batch_size=50,shuffle=True)
#Train(20,train_loader,test_loader,criterion,optimizer,device)

# %%
model = models.vgg16(pretrained=True)
# %%
for params in model.parameters():
    params.requires_grad=False

from collections import OrderedDict

# Build custom classifier
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(p=0.5)),
                                        ('fc2', nn.Linear(5000, 102)),
                                        ('output', nn.LogSoftmax(dim=1))]))

model.classifier = classifier

def validation(model, validateloader, criterion):
    
    val_loss = 0
    accuracy = 0
    
    for images, labels in iter(validateloader):

        images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        val_loss += criterion(output, labels).item()

        probabilities = torch.exp(output)
        
        equality = (labels.data == probabilities.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return val_loss, accuracy

# %%
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

def train_classifier():

    epochs = 20
    steps = 0
    print_every = 40

    model.to('cuda')

    for e in range(epochs):
    
        model.train()

        running_loss = 0

        for images, labels in iter(train_loader):
    
            steps += 1
    
            images, labels = images.to('cuda'), labels.to('cuda')
    
            optimizer.zero_grad()
    
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
            if steps % print_every == 0:
            
                model.eval()
            
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, validate_loader, criterion)
        
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                        "Validation Loss: {:.3f}.. ".format(validation_loss/len(validate_loader)),
                        "Validation Accuracy: {:.3f}".format(accuracy/len(validate_loader)))
        
                running_loss = 0
                model.train()
                    
train_classifier()
# %%
