#%%
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from models import mnistGan as mgan
from logger import Logger as logger
from train import *


DATA_FOLDER = '/root/labs/lab7/VGAN/MNIST'
def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])
        ])
    out_dir = '{}/dataset'.format(DATA_FOLDER)
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

# Load Dataset and attach a DataLoader

data = mnist_data()
print(len(data))
data_loader = torch.utils.data.DataLoader(data, batch_size=100,shuffle=True)
num_batches = len(data_loader)

for i, (x,y) in enumerate(data_loader):
    print(x.shape)
    print(x.view(x.size(0), 784).shape)
    break

# %%
#Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

discriminator = mgan.DiscriminatorNet()
generator = mgan.GeneratorNet()

if torch.cuda.is_available():
    discriminator.to(device)
    generator.to(device)

# %%
##optimizer 
# Optimizers

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Loss function

loss = nn.BCELoss()

# How many epochs to train for

num_epochs = 30

# Number of steps to apply to the discriminator for each step of the generator (1 in Goodfellow et al.)

d_steps = 1
# %%
num_test_samples = 16
test_noise = mgan.noise(num_test_samples)
# %%
trainMnist(data_loader,num_epochs,generator,discriminator,d_optimizer,g_optimizer,loss,num_test_samples,logger) 
# %%
