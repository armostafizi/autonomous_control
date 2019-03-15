from __future__ import print_function
from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg                                                            
import os                                                                                   
import csv                                                                                  
import pandas as pd                                                                         
import numpy as np                                                                          
import sklearn                                                                              
from sklearn.model_selection import train_test_split                                        
from sklearn.utils import shuffle                                                           
                                                                                             
                                                                                            
## 1. Prepare and create generator                                                          
                                                                                            
# Save filepaths of images to `samples` to load into generator                              
samples = []                                                                                
                                                                                            
def add_to_samples(csv_filepath, samples):                                                  
    with open(csv_filepath) as csvfile:                                                     
        reader = csv.reader(csvfile)                                                        
        for line in reader:                                                                 
            samples.append(line)                                                            
    return samples                                                                          
                                                                  
samples = add_to_samples('driving_log.csv', samples)                                   
#samples = add_to_samples('data-recovery-annie/driving_log.csv', samples)
def upd_dir(item):
    updated_item='/nfs/stak/users/estajim/dl/project/images'+item[27:]
    return updated_item                                                                     
# Remove header                            
samples = samples[1:]                                                                  #print(type(samples))                                                                  
#print(samples[0][0][:24])                                                                  
for i in range(len(samples)):                                                
    samples[i][0]=upd_dir(samples[i][0])                                     
    samples[i][1]=upd_dir(samples[i][1])                                     
    samples[i][2]=upd_dir(samples[i][2])                                     
                                                                             
print("Samples: ", len(samples))                                             
# Split samples into training and validation sets to reduce overfitting      
train_samples, validation_samples = train_test_split(samples, test_size=0.1) 
print(np.asarray(train_samples).shape)                                       
#sys.exit()                                                                  


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        
    def forward(self, x):
        x = F.relu(self.conv1(x)) # (32, 32, 32, 32)
	#print(x.data.cpu().numpy().shape)
        x = F.relu(self.conv2(x)) # (32, 32, 32, 32)
	#print(x.data.cpu().numpy().shape)
        x = self.pool(x) # (32, 32, 16, 16)
	#print(x.data.cpu().numpy().shape)
        x = F.relu(self.conv3(x)) # (32, 64, 16, 16)
	#print(x.data.cpu().numpy().shape)
        x = F.relu(self.conv4(x)) # (32, 64, 16, 16)
	#print(x.data.cpu().numpy().shape)
        x = self.pool(x) # (32, 64, 8, 8)
	#print(x.data.cpu().numpy().shape)
        num_features=self.num_flat_features(x)
        x = x.view(-1, self.num_flat_features(x)) # (32, 4096)
	#print(x.data.cpu().numpy().shape)
        x = F.relu(self.fc1(x)) # (32, 512)
	#print(x.data.cpu().numpy().shape)
        x = self.bn(x)
        x = self.fc2(x) # (32, 10)
	#print(x.data.cpu().numpy().shape)
	#print(x.size()[1:])
        return x

    def num_flat_features(self, x):
	#print(x.size()[1:])
        # x.size() is (64L, 8L, 8L)
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
	#print(num_features)
	#sys.exit()
        return num_features

def eval_net(dataloader):
    correct = 0
    total = 0
    total_loss = 0
    net.eval() # Why would I do this?
    criterion = nn.CrossEntropyLoss(size_average=False)
    for data in dataloader:
        images, labels = data
	# My assumption is that images and labes as of now are np arrays.
	# Then, the Variable() funciton makes them ready to be used as inputs of torch
	# Then .cude() enables the GPU computations
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        loss = criterion(outputs, labels)
        total_loss += loss.data[0]
    net.train() # Why would I do this?
    return total_loss / total, correct / total

def plot_hist(train_set,test_set,target):
    epochs=range(1,len(train_set)+1)
    plt.plot(epochs,train_set,'r',label="train "+target)
    plt.plot(epochs,test_set,'g',label="test "+target)
    plt.legend(loc='upper center',shadow=True)
    plt.xlabel("Epochs")
    plt.ylabel(target)
    plt.show()
if __name__ == "__main__":
    BATCH_SIZE = 32 #mini_batch size
    MAX_EPOCH = 10  #maximum epoch to train

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    print(type(trainset))
    #sys.exit()
    trainloader = torch.utils.data.DataLoader(np.asarray(train_samples)[:,0], batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('Building model...')
    net = Net().cuda()
    net.train() # Why would I do this?

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    train_losses=list()
    train_accuracies=list()
    test_losses=list()
    test_accuracies=list()

    print('Start training...')
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            if i % 500 == 499:    # print every 2000 mini-batches
                print('    Step: %5d avg_batch_loss: %.5f' %
                      (i + 1, running_loss / 500))
                running_loss = 0.0
        print('    Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(trainloader)
        test_loss, test_acc = eval_net(testloader)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)

    print('Finished Training')
    print('Saving model...')
    torch.save(net.state_dict(), 'part1.pth')
    plot_hist(train_accuracies,test_accuracies,'Accuracy')
    plot_hist(train_losses,test_losses,'Loss')
