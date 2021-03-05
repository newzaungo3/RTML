import os
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.utils.data as data_utils
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
import alexnet
import torch
import urllib
import os
from models import * 
import torchvision
from PIL import Image
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import model
import pickle
from copy import copy
from copy import deepcopy
preprocess = transforms.Compose(
    [
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


class dogAndMuffin(Dataset):
    def __init__(self,transform=None):
        super().__init__()
        self.datapath = "./labs/dataset/train"
        self.label_path = os.listdir(self.datapath)
        self.label = {'chihuahua':0,'muffin':1}
        self.X = []
        self.y = []
        self.transform = transform
        #loop label values
        for label_name in self.label_path:
            file_dir = os.path.join(self.datapath,label_name)
            filenames = os.listdir(file_dir)
            #loop each file name
            for filename in filenames:
                file = os.path.join(file_dir,filename)
                
                # use io to get image
                image = io.imread(file)
                if self.transform:
                    #perform tranform for each X
                    image = self.transform(image)

                #append to X
                self.X.append(image)
                #appen label to y
                self.y.append(self.label[label_name])

        #self.y = np.array(self.y)
        #self.X = np.array(self.X)
    def __getitem__(self,idx):
        #return X[idx],y[idx]
        return self.X[idx],self.y[idx]
    def __len__(self):
        return len(self.X)

#dataset
train_data = dogAndMuffin(transform=preprocess)

kfold =StratifiedKFold(n_splits=4)
#device
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print('Using device', device)
#model
resnet = ResSENet18()
resnet = resnet.to(device)
#criterion
criterion_2 = nn.CrossEntropyLoss()
params_to_update_2 = resnet.parameters()
optimizer_2 = optim.Adam(params_to_update_2, lr=0.001)

for train_index, test_index in kfold.split(train_data.X, train_data.y):
        tensor_train_x = []
        tensor_valid_x = []
        train_X,train_y = train_data.X[train_index],train_data.y[train_index]
        valid_x,valid_y = train_data.X[test_index],train_data.y[test_index]
        
        '''
        #train_X = preprocess(train_X[0])
        train_y = torch.from_numpy(train_y)
        size = train_X.shape[0]
        for i in range(size):
            trainx =  preprocess(train_X[i])
            tensor_train_x.append(trainx)
        for i in range(valid_x.shape[0]):
            validx = preprocess(valid_x[i])
            tensor_valid_x = np.append(tensor_valid_x,validx)
        #put to dataloader
        '''
        #tensor_train_x = torch.from_numpy(tensor_train_x)
#        train_set = data_utils.TensorDataset(tensor_train_x,train_y)
        #dataloaders ={ 'train': train_loader, 'val': valid_loader }
        #trainmodel with epoch


