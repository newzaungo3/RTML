import os
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.utils.data as data_utils
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
import urllib
import os
from models import * 
import torchvision
from PIL import Image
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import model
import pickle
from copy import copy
from copy import deepcopy

path = "./labs/dataset/train"
test_path = "./labs/dataset/test"
kfold =StratifiedKFold(n_splits=4)
#hyper parameter
batch_size = 10
lr = 0.001
epoch = 25
acc = []
preprocess = transforms.Compose(
    [
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
eval_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


dataset = torchvision.datasets.ImageFolder(root=path, transform=preprocess)
test_dataset = torchvision.datasets.ImageFolder(root=test_path,transform=eval_preprocess)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print('Using device', device)

for fold ,(train_index,valid_index) in enumerate(kfold.split(dataset,dataset.targets)):
    print(f'Fold:{fold}, Train_index:{train_index}, Valid_index:{valid_index}')

    train = torch.utils.data.Subset(dataset, train_index)
    valid = torch.utils.data.Subset(dataset, valid_index)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=1)

    #model
    resnet = ResSENet18()
    resnet = resnet.to(device)
    #loss
    criterion_2 = nn.CrossEntropyLoss()
    params_to_update_2 = resnet.parameters()
    
    if fold == 0:
        optimizer_2 = optim.Adam(params_to_update_2,lr)
    elif fold == 1:
        optimizer_2 = optim.SGD(params_to_update_2,lr)
    elif fold == 2:
        optimizer_2 = optim.Adagrad(params_to_update_2,lr)
    elif fold == 3:
        optimizer_2 = optim.Adamax(params_to_update_2,lr)

    dataloaders = { 'train': train_loader, 'val': valid_loader }
    file = "restSEnet18-chimuff-25epoch"
    
    #train
    best_model2, val_acc_history2, loss_acc_history2,best_acc = model.train_model(resnet, dataloaders, criterion_2, optimizer_2, epoch,file)
    acc.append(best_acc)
    #saving 
    with open(f"/root/labs/loss/{file}-fold{fold}.pk",'wb') as f:
        pickle.dump((val_acc_history2,loss_acc_history2),f)


average = sum(acc)/len(acc)
for i in range(len(acc)):
    print(f"Accuracy fold{i+1}: ",acc[i].item())
print("Average of accuracy is: ",average.item())


#Testing Accuracy
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
correct = 0
total = 0
with torch.no_grad():
    for data in tqdm(test_loader,desc="Testing"):
        images, labels = data[0].to(device), data[1].to(device)
        outputs = best_model2(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
print('Accuracy on test images: %d %%' % (100 * correct / total))