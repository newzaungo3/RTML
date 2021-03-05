import torch
import urllib
import os
import torchvision
from PIL import Image
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Tensorboard
logdir = '/root/labs/reports'
#writer = SummaryWriter(log_dir=logdir)


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    
    ]
)
#Hyperparams
# Hyper-parameters 
input_size = 784
hidden_size = 89
num_classes = 10
num_epochs = 25
batch_size = 300
learning_rate = 0.001
momentum= 0.9

#Download CIFAR10
train_dataset = torchvision.datasets.CIFAR10(root='/root/labs/data', 
                                           train=True, 
                                           transform=transform,  
                                           download=True)
test_dataset = torchvision.datasets.CIFAR10(root='/root/labs/data', 
                                          train=False, 
                                          transform=transform)
#split train and validate set
train_size = int(len(train_dataset)*0.8)
valid_size = int(len(train_dataset)*0.2)

train_set , valid_set = torch.utils.data.random_split(train_dataset,[train_size,valid_size])


#Dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                           batch_size=batch_size, 
                                           shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=valid_set, 
                                           batch_size=batch_size, 
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

#Alexnet
AlexNet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
#Update network to classify 10 output
AlexNet_model.classifier[4] = torch.nn.Linear(4096,1024)
AlexNet_model.classifier[6] = torch.nn.Linear(1024,10)
print(AlexNet_model.eval())
AlexNet_model.to(device)

#Loss
criterion = nn.CrossEntropyLoss()
#Optimizer(SGD)
optimizer = optim.SGD(AlexNet_model.parameters(), lr=learning_rate, momentum=momentum)

#Training
total_step = len(train_loader)
train_loss = []
valid_loss = []
train_accuracy = []
valid_accuracy = []

for epoch in range(num_epochs):  # loop over the dataset multiple times
    iter_loss = 0.0
    correct = 0
    iterations = 0
    total = 0
    #Trainset
    for i, data in enumerate(tqdm(train_loader), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = AlexNet_model(inputs)
        loss = criterion(output, labels)
        iter_loss += loss.item()
        #write report
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        optimizer.step()

        #Record output and accuracy
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        iterations += 1
        correct += (predicted == labels).sum().item()


    train_loss.append(iter_loss/iterations)
    train_accuracy.append(100 * correct / total)
        
    #Validate set
    iterloss_val = 0.0
    correct_val = 0
    iterations = 0
    total = 0
    for i, data in enumerate(tqdm(valid_loader), 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        output = AlexNet_model(inputs)
        loss_val= criterion(output, labels)
        iterloss_val += loss_val.item()
        #write report
        writer.add_scalar("Loss/val", loss_val, epoch)

        #Record output and accuracy
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct_val += (predicted == labels).sum().item()
        iterations += 1
    valid_loss.append(iterloss_val/iterations)
    valid_accuracy.append(100 * correct_val / total)

    print ('Epoch [{}/{}],Train_loss: {:.4f},Train_acc{:.4f},Valid_loss{:.4f},Valid_acc{:.4f}' 
            .format(epoch+1, num_epochs, train_loss[-1], train_accuracy[-1], 
             valid_loss[-1], valid_accuracy[-1]))

print('Finished Training of AlexNet')
print("**Testing**")
#Testing Accuracy
correct = 0
total = 0
with torch.no_grad():
    for data in tqdm(test_loader):
        images, labels = data[0].to(device), data[1].to(device)
        outputs = AlexNet_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
print('Accuracy on test images: %d %%' % (100 * correct / total))
writer.flush()
writer.close()

# Save the model checkpoint
torch.save(AlexNet_model.state_dict(),f"/root/labs/checkpoints/alexnet-cifar-10-{num_epochs}-Train_acc:{train_accuracy[-1]:.2f}%-test acc:{acc}-epochs-sgd-0.02.pth" )
