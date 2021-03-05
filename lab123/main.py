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
#hyperparam
batch_size = 4
epoch = 25

#Data augmentation
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

#Download CIFAR10
full_train_dataset = torchvision.datasets.CIFAR10(root='/root/labs/data', 
                                           train=True, 
                                           download=True)


#split train , validate set = 
train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [40000, 10000])
train_dataset.dataset = copy(full_train_dataset)
train_dataset.dataset.transform = preprocess

val_dataset.dataset.transform = eval_preprocess
print(type(train_dataset.dataset.data))

test_dataset = torchvision.datasets.CIFAR10(root='/root/labs/data', 
                                          train=False, 
                                          transform=eval_preprocess)

#Dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           num_workers=4)

valid_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False,
                                           num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False,num_workers=4)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print('Using device', device)

resnet = ResSENet18()
resnet = resnet.to(device)
# Make optimizer and Loss function
criterion_2 = nn.CrossEntropyLoss()
params_to_update_2 = resnet.parameters()
optimizer_2 = optim.Adam(params_to_update_2, lr=0.001)

dataloaders = { 'train': train_loader, 'val': valid_loader }
print(resnet.eval())
file = "restSEnet18-25epoch"
best_model2, val_acc_history2, loss_acc_history2 = model.train_model(resnet, dataloaders, criterion_2, optimizer_2, epoch,file)

#saving 
with open(f"/root/labs/loss/{file}.pk",'wb') as f:
    pickle.dump((val_acc_history2,loss_acc_history2),f)

#Test model
#Testing Accuracy
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