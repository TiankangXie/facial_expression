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
from test_model import toy_Net
from sklearn.metrics import f1_score
from toy_vgg import toy_VGG
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
def train_val_dataset(dataset,val_split = 0.25):
    train_idx,val_idx=train_test_split(list(range(len(dataset))),test_size = 0.25)
    datasets = {}
    datasets['train'] = Subset(dataset,train_idx)
    datasets['test'] = Subset(dataset,val_idx)
    return(datasets)


# %%
transform = transforms.Compose([
            transforms.Resize(257),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
Dataset01 = image_Loader(csv_dir="F:\\here.csv",img_dir="F:\\FaceExprDecode\\F001\\",transform=transform, action_unit='6')
train_set,test_set = torch.utils.data.random_split(Dataset01,[1000,230])
train_loader = DataLoader(dataset=train_set,batch_size=50,shuffle=True)
test_loader = DataLoader(dataset=test_set,batch_size=50,shuffle=True)
#Train(20,train_loader,test_loader,criterion,optimizer,device)

# %%
net01 = toy_Net()
net01.to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(net01.parameters(),lr = 0.001)

for epoch in range(5):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = net01(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        
        if i % 500 == 499:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
# %%
correct = 0
total = 0

with torch.no_grad():
    for images,labels in test_loader:
        images, labels = images.to(device),labels.to(device)

        outputs = net01(images)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        
        correct += (predicted == labels).sum().item()
        print(labels.cpu())
        print(predicted.cpu())
        f1_batch = f1_score(labels.cpu(),predicted.cpu(),average='macro')
        print(f1_batch)


# %%
vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

# %%
def get_vgg_layers(config, batch_norm):
    
    layers = []
    in_channels = 3
    
    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size = 2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size = 3, padding = 1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace = True)]
            else:
                layers += [conv2d, nn.ReLU(inplace = True)]
            in_channels = c
            
    return nn.Sequential(*layers)
vgg11_layers = get_vgg_layers(vgg11_config, batch_norm = True)
print(vgg11_layers)
# %%
OUTPUT_DIM = 7
model = toy_VGG(vgg11_layers,OUTPUT_DIM)
print(model)
# IN_FEATURES = pretrained_model.classifier[-1].in_features 
# final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
# pretrained_model.classifier[-1] = final_fc
# print(pretrained_model.classifier)

# %%
import torchvision.models as models
pretrained_model = models.vgg11_bn(pretrained=True)

# %%
pretrained_model.classifier[-1]

# %%
IN_FEATURES = pretrained_model.classifier[-1].in_features 

final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
# %%
pretrained_model.classifier[-1] = final_fc

# %%
print(pretrained_model.classifier)

# %%
model.load_state_dict(pretrained_model.state_dict())

# %%
for parameter in model.classifier[:-1].parameters():
    parameter.requires_grad = False

# %%    
START_LR = 1e-7

optimizer = optim.Adam(model.parameters(), lr = START_LR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)
# %%
from LRFinder import LRFinder
import torch
END_LR=10
NUM_ITER=100

lr_finder = LRFinder(model,optimizer,criterion,device)
lrs,losses = lr_finder.range_test(train_loader,END_LR,NUM_ITER)
# %%

def plot_lr_finder(lrs, losses, skip_start = 5, skip_end = 5):
    
    if skip_end == 0:
        lrs = lrs[skip_start:]
        losses = losses[skip_start:]
    else:
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]
    
    fig = plt.figure(figsize = (16,8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(lrs, losses)
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Loss')
    ax.grid(True, 'both', 'x')
    plt.show()
plot_lr_finder(lrs, losses, skip_start = 10, skip_end = 20)

# %%
