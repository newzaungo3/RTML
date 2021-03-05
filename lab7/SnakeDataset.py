from torch.utils.data import Dataset
import math
import numpy as np
import torch

class snakeDataset(Dataset):
    def __init__(self,numsample):
        super().__init__()
        self.pi = np.pi
        self.numsample = numsample
        self.result = torch.zeros([numsample,2])

        for i in range(self.numsample):
            self.r = torch.randn(1)
            self.theta = torch.FloatTensor(1).uniform_(0,2 * self.pi)
            if 0.5*self.pi <= self.theta.item() and self.theta.item() <= 1.5*self.pi :
                self.result[i,0] = (10+self.r.item()) * math.cos(self.theta.item())
                self.result[i,1] = ((10+self.r.item()) * math.sin(self.theta.item())) + 10
            else:
                self.result[i,0] = (10+self.r.item()) * math.cos(self.theta.item())
                self.result[i,1] = ((10+self.r.item()) * math.sin(self.theta.item())) - 10           

    def  __getitem__(self,idx):
        return self.result[idx]

    
    def __len__(self):
        return len(self.result)