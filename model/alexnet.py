import torch
import torch.nn as nn
import torchvision
class AlexNet32(nn.Module):
    def __init__(self,classes) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3,96,3,1,1,bias=False)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(96,256,3,1,1,bias=False)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(256,384,3,1,1,bias=False)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(384,384,3,1,1,bias=False)
        self.relu4 = nn.ReLU()
        
        self.conv5 = nn.Conv2d(384,256,3,1,1,bias=False)
        self.relu5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2,2)

        self.fc1   = nn.Linear(256*4*4,4096,bias=False)
        self.relu6 = nn.ReLU()
        self.fc2   = nn.Linear(4096,4096,bias=False)
        self.relu7 = nn.ReLU()
        self.fc3   = nn.Linear(4096,classes)
    
    def forward(self,t):
        t = self.conv1(t)
        t = self.relu1(t)
        t = self.pool1(t)

        t = self.conv2(t)
        t = self.relu2(t)
        t = self.pool2(t)

        t = self.conv3(t)
        t = self.relu3(t)
        t = self.conv4(t)
        t = self.relu4(t)
        t = self.conv5(t)
        t = self.relu5(t)
        t = self.pool3(t)
        
        t = self.fc1(t.reshape(-1,256*4*4))
        t = self.relu6(t)
        t = self.fc2(t)
        t = self.relu7(t)
        t = self.fc3(t)
        return t