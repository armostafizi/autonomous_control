from __future__ import print_function
from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import pandas as pd                                                                         
import numpy as np        
import cv2
import csv
import sys
import os
from datetime import datetime

# load the samples and split them into training and validation sets
def read_samples(csv_filepath, validation_per = 0.2):
    samples = []
    with open(csv_filepath) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line in reader:                                                                 
            samples.append(line)
    validation_count = int(validation_per * len(samples))
    training_count = len(samples) - validation_count
    training_samples, validation_samples = random_split(samples,\
                                                        lengths = [training_count, validation_count])
    return training_samples, validation_samples 

def augment(image_dir, imgName, angle):
    name = image_dir + imgName.split('/')[-1]
    current_image = cv2.imread(name)
    current_image = current_image[65:-25, :, :]
    if np.random.rand() < 0.5:
        current_image = cv2.flip(current_image, 1)
        angle = angle * -1.0  
    return current_image, angle


class Dataset(data.Dataset):
    def __init__(self, samples, image_dir, transform=None):
        self.samples = samples
        self.image_dir = image_dir
        self.transform = transform
    
    def __getitem__(self, index):
        batch_samples = self.samples[index]
        steering_angle = float(batch_samples[3])
        center_img, steering_angle_center = augment(self.image_dir, batch_samples[0], steering_angle)
        left_img, steering_angle_left = augment(self.image_dir, batch_samples[1], steering_angle + 0.4)
        right_img, steering_angle_right = augment(self.image_dir, batch_samples[2], steering_angle - 0.4)
        center_img = self.transform(center_img)
        left_img = self.transform(left_img)
        right_img = self.transform(right_img)
        return (center_img, float(steering_angle_center)),\
               (left_img, float(steering_angle_left)),\
               (right_img, float(steering_angle_right))
               
    def __len__(self):
        return len(self.samples)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(87040, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        num_features=self.num_flat_features(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def eval_net(dataloader):
    correct = 0
    total = 0
    total_loss = 0
    net.eval() # Why would I do this?
    criterion = nn.MSELoss(reduction='sum')
    for center, left, right in dataloader:
        images, targets = center
        images, targets = Variable(images).cuda(), Variable(targets.float()).cuda().unsqueeze(1)
        outputs = net(images)
        #_, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        #correct += (predicted == targets.data).sum()
        loss = criterion(outputs, targets)
        total_loss += loss.data.item()
    net.train() # Why would I do this?
    return total_loss / total


if __name__ == "__main__":
    BATCH_SIZE = 32 #mini_batch size
    MAX_EPOCH = 2  #maximum epoch to train
   
    data_dir = sys.argv[1] # data directory

    train_samples, test_samples = read_samples(data_dir + 'driving_log.csv')                     
     
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = Dataset(train_samples, data_dir + 'IMG/', transform)
    test_set = Dataset(test_samples, data_dir + 'IMG/', transform)

    trainloader = DataLoader(train_set,\
                             batch_size=BATCH_SIZE,\
                             shuffle=True,\
                             num_workers=4)
    testloader = DataLoader(test_set,\
                            batch_size=BATCH_SIZE,\
                            shuffle=False,\
                            num_workers=4)


    print('Building model...')
    net = Net().cuda()
    net.train() # Why would I do this?

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    train_losses=list()
    test_losses=list()

    print('Start training...')
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (center, left, right) in enumerate(trainloader, 0):
            # get the inputs
            inputs, targets = center # only use center for now

            # wrap them in Variable
            inputs, targets = Variable(inputs).cuda(), Variable(targets.float()).cuda().unsqueeze(1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('    Step: %5d avg_batch_loss: %.5f' %
                      (i + 1, running_loss / 500))
                running_loss = 0.0
        print('    Finish training this EPOCH, start evaluating...')
        train_loss = eval_net(trainloader)
        test_loss = eval_net(testloader)
        print('EPOCH: %d train_loss: %.5f test_loss: %.5f' %
              (epoch+1, train_loss, test_loss))
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    print('Finished Training')
    plt.plot(train_losses, label='train_loss')
    plt.plot(test_losses, label='test_loss')
    plt.legend()
    now_str = datetime.now().strftime("%d_%s")
    plt.savefig('models/model_%s_%.3f.png' % (now_str, min(test_losses)))
    print('Saving model...')
    torch.save(net.state_dict(), 'models/model_%s_%.3f.pth' % (now_str, min(test_losses)))
