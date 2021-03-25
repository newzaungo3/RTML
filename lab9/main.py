from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from models import vae as vae
from process import *
from tqdm import tqdm
import utils
import visdom

if __name__ == '__main__':

    #hyperparameter
    dataname ='mnist'
    seed = 1
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size=128

    '''
    #FOR MNIST DATASET
    out_dir = './torch_data/VGAN/MNIST/dataset'
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(root=out_dir, download=True, train=True, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST(root=out_dir, train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    '''
    #For ait-celeb dataset
    
    compose = transforms.Compose(
    [
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ])

    ds = ImageFolder(root='dataset/', transform=compose)

    ratio = [int(len(ds)*0.7), len(ds) - int(len(ds)*0.7)]

    train_dataset, test_dataset = torch.utils.data.random_split(ds, ratio)

    batch_size=4

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                shuffle=False,num_workers=1, pin_memory=True)
    print('train_loader', len(train_loader))
    print('test_loader', len(test_loader))
    
    model = vae.VAE()
    if torch.cuda.is_available():
        print("I'am using GPU :)")
        model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas = (0.9,0.9), weight_decay = 0.0005)
    epochs = 100

    viz = visdom.Visdom()
    plotter = utils.VisdomLinePlotter(env_name='main')
    sample_image = utils.VisdomImage(env_name='main')
    recon = utils.VisdomImage(env_name='main')
    
    for epoch in range(1, epochs + 1):        

        with torch.no_grad():
            sample = torch.randn(32,32).to(device)
            sample = model.decode(sample).cpu()
            print("save image: " + './results/'+str(dataname) +'/sample/sample_' + str(epoch) + '.jpg')
            save_image(sample,'./results/'+ str(dataname) +'/sample/sample_' + str(epoch) + '.jpg')
            sample_image.display_image(sample, 0, 'SAMPLE RECON')
        model = train(model,device,train_loader,optimizer,epoch, plotter)
        test(model,device,test_loader,epoch, plotter,recon)