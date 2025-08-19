import torch
import sys
sys.path.append("..")
from model import ConvNet_cifar10
from torchvision import transforms
from PIL import Image
from importlib import import_module
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
net_name = 'AlexNet_cifar10'
pic_path = r'../cifar-10/train/13.png'

net_module = import_module("model."+net_name)
net = getattr(net_module,net_name)

network = net()
network.load_state_dict(torch.load('../model/'+net_name+'.pth'))
network = network.to(device)

data_trans = transforms.Compose([transforms.ToTensor()])

img = Image.open(pic_path)
img = data_trans(img)
img = torch.unsqueeze(img,dim=0).to(device)
network.eval()
with torch.no_grad():
    predict = network(img)

print(predict)