import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.nn.init as init

class DELVgg1632(nn.Module):
    def __init__(self,nums):
        super().__init__()
        self.nums = nums
        #input 3*32*32
        self.conv1 = nn.Conv2d(3,64,3,1,1,bias=False)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64,64,3,1,1,bias=False)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)
        # 64*32*32
        self.conv3 = nn.Conv2d(64,128,3,1,1,bias=False)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128,128,3,1,1,bias=False)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)
        #128*16*16
        self.conv5 = nn.Conv2d(128,256,3,1,1,bias=False)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256,256,3,1,1,bias=False)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(256,256,3,1,1,bias=False)
        self.relu7 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2,2)
        #256*8*8
        self.conv8 = nn.Conv2d(256,512,3,1,1,bias=False)
        self.relu8 = nn.ReLU()
        self.conv9 = nn.Conv2d(512,512,3,1,1,bias=False)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(512,512,3,1,1,bias=False)
        self.relu10 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2,2)
        #512*4*4
        self.conv11 = nn.Conv2d(512,512,3,1,1,bias=False)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(512,512,3,1,1,bias=False)
        self.relu12 = nn.ReLU()
        self.conv13 = nn.Conv2d(512,512,3,1,1,bias=False)
        self.relu13 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2,2)
        #512*2*2,4096
        self.linear1 = nn.Linear(512,4096,bias=False)
        self.relu14 = nn.ReLU()
        self.linear2 = nn.Linear(4096,4096,bias=False)
        self.relu15 = nn.ReLU()
        self.linear3 = nn.Linear(4096,self.nums,bias=False)
        # self.relu16 = nn.ReLU()
        self.get_weight()

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.pool3(x)

        x = self.conv8(x)
        x = self.relu8(x)
        x = self.conv9(x)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.relu10(x)
        x = self.pool4(x)

        x = self.conv11(x)
        x = self.relu11(x)
        x = self.conv12(x)
        x = self.relu12(x)
        x = self.conv13(x)
        x = self.relu13(x)
        x = self.pool5(x)

        x = x.view(x.size(0),-1)
        x = self.linear1(x)
        x = self.relu14(x)
        x = self.linear2(x)
        x = self.relu15(x)
        x = self.linear3(x)
        # x = self.relu16(x)

        return x
    
    def get_weight(self):
        net_torch = models.vgg16(pretrained=True)
        print(net_torch)
        self.conv1.weight.data = net_torch.features[0].weight.data
        # self.conv1.bias.data = net_torch.features[0].bias.data

        self.conv2.weight.data = net_torch.features[2].weight.data
        # self.conv2.bias.data = net_torch.features[2].bias.data

        self.conv3.weight.data = net_torch.features[5].weight.data
        # self.conv3.bias.data = net_torch.features[5].bias.data

        self.conv4.weight.data = net_torch.features[7].weight.data
        # self.conv4.bias.data = net_torch.features[7].bias.data

        self.conv5.weight.data = net_torch.features[10].weight.data
        # self.conv5.bias.data = net_torch.features[10].bias.data

        self.conv6.weight.data = net_torch.features[12].weight.data
        # self.conv6.bias.data = net_torch.features[12].bias.data

        self.conv7.weight.data = net_torch.features[14].weight.data
        # self.conv7.bias.data = net_torch.features[14].bias.data

        self.conv8.weight.data = net_torch.features[17].weight.data
        # self.conv8.bias.data = net_torch.features[17].bias.data

        self.conv9.weight.data = net_torch.features[19].weight.data
        # self.conv9.bias.data = net_torch.features[19].bias.data

        self.conv10.weight.data = net_torch.features[21].weight.data
        # self.conv10.bias.data = net_torch.features[21].bias.data

        # self.conv11.weight.data = net_torch.features[24].weight.data
        # # self.conv11.bias.data = net_torch.features[24].bias.data

        # self.conv12.weight.data = net_torch.features[26].weight.data
        # # self.conv12.bias.data = net_torch.features[26].bias.data

        # self.conv13.weight.data = net_torch.features[28].weight.data
        # self.conv13.bias.data = net_torch.features[28].bias.data

        # 第一个线性层不载入 使用凯明初始化
        # init.kaiming_normal_(self.linear1.weight,nonlinearity='relu')
        # self.linear1.bias.data = net_torch.classifier[0].bias.data

        # self.linear2.weight.data = net_torch.classifier[3].weight.data
        # self.linear2.bias.data = net_torch.classifier[3].bias.data

        # self.linear3.weight.data = net_torch.classifier[6].weight.data
        # self.linear3.bias.data = net_torch.classifier[6].bias.data




class Vgg1132(nn.Module):
    def __init__(self,nums):
        super().__init__()
        self.nums = nums
        #input 3*32*32
        self.conv1 = nn.Conv2d(3,64,3,1,1,bias=False)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)
        # 64*32*32
        self.conv2 = nn.Conv2d(64,128,3,1,1,bias=False)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)
        #128*16*16
        self.conv3 = nn.Conv2d(128,256,3,1,1,bias=False)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256,256,3,1,1,bias=False)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2,2)
        #256*8*8
        self.conv5 = nn.Conv2d(256,512,3,1,1,bias=False)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(512,512,3,1,1,bias=False)
        self.relu6 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2,2)
        #512*4*4
        self.conv7 = nn.Conv2d(512,512,3,1,1,bias=False)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(512,512,3,1,1,bias=False)
        self.relu8 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2,2)
        #512*2*2,4096
        self.linear1 = nn.Linear(512,4096,bias=False)
        self.relu9 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(4096,4096,bias=False)
        self.relu10 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear3 = nn.Linear(4096,self.nums,bias=False)
        # self.relu11 = nn.ReLU()
        self.get_weight()

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool3(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool4(x)

        x = self.conv7(x)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.relu8(x)
        x = self.pool5(x)

        x = x.view(x.size(0),-1)
        x = self.linear1(x)
        x = self.relu9(x)
        x = self.linear2(x)
        x = self.relu10(x)
        x = self.linear3(x)
        # x = self.relu11(x)

        return x
    
    def get_weight(self):
        net_torch = models.vgg11(pretrained=True)
        # print(net_torch)
        self.conv1.weight.data = net_torch.features[0].weight.data
        # self.conv1.bias.data = net_torch.features[0].bias.data

        self.conv2.weight.data = net_torch.features[3].weight.data
        # self.conv2.bias.data = net_torch.features[3].bias.data

        self.conv3.weight.data = net_torch.features[6].weight.data
        # self.conv3.bias.data = net_torch.features[6].bias.data

        self.conv4.weight.data = net_torch.features[8].weight.data
        # self.conv4.bias.data = net_torch.features[8].bias.data

        self.conv5.weight.data = net_torch.features[11].weight.data
        # self.conv5.bias.data = net_torch.features[11].bias.data

        self.conv6.weight.data = net_torch.features[13].weight.data
        # self.conv6.bias.data = net_torch.features[13].bias.data

        self.conv7.weight.data = net_torch.features[16].weight.data
        # self.conv7.bias.data = net_torch.features[16].bias.data

        self.conv8.weight.data = net_torch.features[18].weight.data
        # self.conv8.bias.data = net_torch.features[18].bias.data


        # 第一个线性层不载入 使用凯明初始化
        # init.kaiming_normal_(self.linear1.weight,nonlinearity='relu')
        # self.linear1.bias.data = net_torch.classifier[0].bias.data

        # self.linear2.weight.data = net_torch.classifier[3].weight.data
        # self.linear2.bias.data = net_torch.classifier[3].bias.data

        # self.linear3.weight.data = net_torch.classifier[6].weight.data
        # self.linear3.bias.data = net_torch.classifier[6].bias.data