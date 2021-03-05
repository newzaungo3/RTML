#%% 
from SnakeDataset import *
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from models import vanillaGan
from logger import Logger as logger
from train import *
import matplotlib.pyplot as plt
import torch.nn as nn
#dataset
data = snakeDataset(10000)
# %%

for i in range(1000):
    plt.scatter(data[i][0],data[i][1])
plt.show()

# %%
data_loader = torch.utils.data.DataLoader(data, batch_size=100,shuffle=True)
num_batches = len(data_loader)

# %%
print(num_batches)
# %%
#Device
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

discriminator = vanillaGan.DiscriminatorNet()
generator = vanillaGan.GeneratorNet()

if torch.cuda.is_available():
    discriminator.to(device)
    generator.to(device)


# %%
# Optimizers

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function

loss = nn.BCELoss()

# Number of epochs of training
num_epochs = 500

# Number of steps to apply to the discriminator for each step of the generator (1 in Goodfellow et al.)

d_steps = 1
# %%
num_test_samples = 16

# %%
trainSnake(data_loader,num_epochs,generator,discriminator,d_optimizer,g_optimizer,loss,num_test_samples,logger)
