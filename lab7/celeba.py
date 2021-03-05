#%%
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from models import dcGan
from logger import Logger as logger
from train import *
import matplotlib.pyplot as plt
import torch.nn as nn

#%%
path = "/root/labs/lab7/CelebaData"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
compose = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])

train_data = datasets.ImageFolder(path,transform=compose)
batch_size = 100

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,
                                           batch_size=batch_size, num_workers=4)
#%%
for i ,(real,_) in enumerate(train_loader):
    print(real.shape)
    break
# %%
# Custom weight initialization

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)

# Instantiate networks

generator = dcGan.GenerativeNet()
generator.apply(init_weights)
discriminator = dcGan.DiscriminativeNet()
discriminator.apply(init_weights)

# Enable cuda if available

if torch.cuda.is_available():
    generator.to(device)
    discriminator.to(device)

#%%
# Optimizers

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function

loss = nn.BCELoss()

# Number of epochs of training
num_epochs = 50

#%%
num_test_samples = 16
# %%
trainDC(train_loader,num_epochs,generator,discriminator,d_optimizer,g_optimizer,loss,num_test_samples,logger) 
# %%
