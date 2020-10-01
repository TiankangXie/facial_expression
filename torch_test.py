# %%
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import PIL.Image
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
    def concat_csv(self):
        appended_data = []
        for task_i in self.tasks:
            data = pd.read_csv(self.csv_file_path+"F001_"+task_i+".csv",header=0)
            data['subject']=task_i
            appended_data.append(data)
        appended_data = pd.concat(appended_data)
        self.csv_file = appended_data

            def find_pics_from_csv(self):

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
train_idx,val_idx=train_test_split(list(range(f01_csv.shape[0])),test_size = 0.30)
# %%
