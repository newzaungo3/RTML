import urllib
import os
from torch.utils.tensorboard import SummaryWriter



#%%
import pickle
acc1,loss1= pickle.load(open("/root/labs/loss/restSEnet18-chimuff-25epoch-fold0.pk",'rb'))
acc2,loss2= pickle.load(open("/root/labs/loss/restSEnet18-chimuff-25epoch-fold1.pk",'rb'))
acc3,loss3= pickle.load(open("/root/labs/loss/restSEnet18-chimuff-25epoch-fold2.pk",'rb'))
acc4,loss4= pickle.load(open("/root/labs/loss/restSEnet18-chimuff-25epoch-fold3.pk",'rb'))

res_acc , res_loss = pickle.load(open("/root/labs/loss/restnet18-25epoch.pk",'rb'))
se_acc,se_loss = pickle.load(open("/root/labs/loss/restSEnet18-25epoch.pk",'rb'))




# %%
from torch.utils.tensorboard import SummaryWriter
logdir = '/root/labs/reports'
writer = SummaryWriter(log_dir=logdir)
epoch = 25
acc = []
for i in range(epoch):
    writer.add_scalars('Accuracy', {'fold1':acc1[i].item(),
                                    'fold2':acc2[i].item(),
                                    'fold3': acc3[i].item(),
                                    'fold4': acc4[i].item()}, i)
    writer.add_scalars('Loss', {'fold1':loss1[i],
                                    'fold2':loss2[i],
                                    'fold3': loss3[i],
                                    'fold4': loss4[i]}, i)
writer.flush()
writer.close()

#%%
# %%
from torch.utils.tensorboard import SummaryWriter
logdir = '/root/labs/reports'
writer = SummaryWriter(log_dir=logdir)
epoch = 25
acc = []
for i in range(epoch):
    writer.add_scalars('Accuracy RestNet & SENet', {'RestNet':res_acc[i].item(),
                                    'RestSEnet':se_acc[i].item()}, i)
    writer.add_scalars('Loss RestNet & SENet', {'RestNet':loss1[i],
                                    'RestSEnet':loss2[i]}, i)
writer.flush()
writer.close()
# %%
