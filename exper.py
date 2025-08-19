"""
This script is designed to provide interpretability of convolutional neural networks (CNNs)  
from multiple perspectives. By analyzing the internal mechanisms and learned representations,  
it aims to enhance understanding of CNN decision processes and reveal insights into feature  
extraction, hierarchical abstraction, and model behavior.
"""


import numpy as np
import os
import numpy as np
import torch
import argparse
# import matplotlib.pyplot as plt
import sys
# from dataset import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
from function.quanindex import quanindex
from tqdm import tqdm
from dataset import *
import csv
import gc
from sklearn.cluster import DBSCAN
import numpy as np
from model.alexnet import AlexNet32
from model.vgg import *
from model.newmodel import *


parser = argparse.ArgumentParser()

parser.add_argument('--device', '-d', type=str, default='5',
                    help='cuda number')
parser.add_argument('--acc', '-ac', type=int, default='90',
                    help='model accuracy')
parser.add_argument('--mode', type=str, default='Vgg1132_tinyimg',
                    help='Alexnet32_tinyimg Vgg1132_tinyimg Vgg1632_tinyimg Alexnet32_imagenet')
parser.add_argument('--datamode', type=str, default='tinyimg32',
                    help='imagenet32 tinyimg32')
parser.add_argument('--running', type=str, default='simi',
                    help='simi draw')
parser.add_argument('--layer', type=int, default='4',
                    help='')
args = parser.parse_args()

device = torch.device("cuda:"+args.device if torch.cuda.is_available() else "cpu")

ac = args.acc 

model_name = args.mode

DATA = 'tinyimg32'
class_name = '1'

# imageten = 'tem0/0000.pt'
xmldir = '/nas/douhui/douhui/dataset/tiny-imagenet-5/bboxorg/n01443537_boxes.txt'
imageten = '/nas/douhui/douhui/scandata/'+ model_name+'_'+str(ac)+'/'+class_name+'/tem0/0.pt'
scandir =  '/nas/douhui/douhui/scandata/'+ model_name+'_'+str(ac)+'/'+class_name+'/scan_ten'
featuredir = '/nas/douhui/douhui/scandata/'+ model_name+'_'+str(ac)+'/'+class_name+'/scan_fea'



if model_name == 'Alexnet32_tinyimg':
    layerLst = ['c1','r1','p1','c2','r2','p2','c3','r3','c4','r4','c5','r5','p5','fc1','r6','fc2','r7','fc3']
elif model_name == 'Vgg1132_tinyimg':
    layerLst = ['c1','r1','p1','c2','r2','p2','c3','r3','c4','r4','p3','c5','r5','c6','r6','p4'
                    ,'c7','r7','c8','r8','p5','fc1','r9','fc2','r10','fc3']
elif model_name == 'Vgg1632_tinyimg':
    layerLst = ['c1','r1','c2','r2','p1','c3','r3','c4','r4','p2','c5','r5','c6','r6'
                    ,'c7','r7','p3','c8','r8','c9','r9','c10','r10','p4','c11','r11','c12','r12','c13','r13','p5'
                    ,'fc1','r14','fc2','r15','fc3']
    
laynum = len(layerLst)
layerNameLst = [f"{i:02d}" for i in range(laynum)]

tensordir = '/nas/douhui/douhui/quantify_NNScanner/tensor/'

savedir = '/nas/douhui/douhui/quantify_NNScanner/exp_paper/'

dir = savedir+args.mode+'_'+str(args.acc)+'/'
if not os.path.exists(dir):
    os.makedirs(dir)
name = '1_'+args.mode+'_'+str(args.acc)+'.pt'
lst = torch.load(tensordir+name)

def exp1():
    
    title = ['ssim','l','c','s','mu','sigma','sum_map','ef','no','all']

    #SSIM
    print('ssim**********')
    averages1 = [np.average(Variable(l1[:,0]).data.numpy()) for l1 in lst]
    print(averages1)


    print('fq**********')
    #fq
    averages2 = [np.average(Variable(l2[:,6]*l2[:,7]*l2[:,8]*1000).data.numpy()) for l2 in lst]
    print(averages2)
    for j2 in range(len(lst)):
        l2 = lst[j2][:,6]*lst[j2][:,7]*lst[j2][:,8]
        x2 = np.average(Variable(l2).data.numpy())
        y2 = layerNameLst[j2]
        print(str(x2*1000))

    print('layername**********')
    for j3 in range(len(lst)):
        y3 = layerLst[j3]
        print(str(y3))



def exp2():

    
    X = [Variable(l2[:,6]*l2[:,7]*l2[:,8]).data.numpy() for l2 in lst][22]
    Y = np.zeros(len(X))
    coordinates = [(x, y) for x, y in zip(X, Y)]

    # DBSCAN
    dbscan = DBSCAN(eps=0.00003, min_samples=6)

    # cluster
    labels = dbscan.fit_predict(coordinates)
    print(max(labels))
    s = sum(1 for elem in labels if elem == -1)

    print(str(s)+'/'+str(len(labels)))



def exp3():

    for j in (range(len(lst))):
        layername = layerNameLst[j]
        featureTensor = torch.load(featuredir +'/'+ layername +'/tensor.pt')
        if len(featureTensor.size())==4:
            featureTensor = torch.flatten(featureTensor,2,3)
            featureTensor = torch.flatten(featureTensor,1,2)
            # bs = featureTensor.shape[1]*featureTensor.shape[2]
        x = Variable(featureTensor[0]).data.numpy()
        s = lst[j][:,8]
        y = Variable(s).data.numpy()

        # Pearson correlation coefficient
        corr_coef = np.corrcoef(x, y)[0, 1]
        print(layerLst[j]+"_Pearson correlation coefficient:", corr_coef)




def exp4():
    if model_name == 'Vgg1132_tinyimg':
        classes = 5
        modeldir = '/nas/douhui/douhui/trained_model/tiny200ILSVRC_vgg1132_without_pretraintest/vgg1132'+'.pth'
        model = Vgg1132(classes)
        mod = "vgg1132" #alexnet32，vgg1132, vgg1632
    if model_name == 'Vgg1632_tinyimg':
        classes = 5
        modeldir = '/nas/douhui/douhui/trained_model/tiny200ILSVRC_vgg1632_without_pretraintest/vgg1632'+'.pth'
        model = Vgg1632(classes)
        mod = "vgg1632" #alexnet32，vgg1132, vgg1632
    if model_name == 'Alexnet32_tinyimg':
        classes = 5
        modeldir = '/nas/douhui/douhui/trained_model/tiny200ILSVRC_alexnet32_without_pretraintest/alexnet32'+'.pth'
        model = AlexNet32(classes)
        mod = "alexnet32" #alexnet32，vgg1132, vgg1632

    load_checkpoint(modeldir, model)
    model = model.to(device)
    print(model)

    valdir = '/nas/douhui/douhui/dataset/tiny-imagenet-5/val'
    val_loader = torch.utils.data.DataLoader(SigmaDataSet(valdir, model=mod), batch_size=500, shuffle=True, num_workers=8, pin_memory=True)
    # validate(val_loader, model)


    model1 = DELVgg1632(classes)
    model1 = model1.to(device)
    validate(val_loader, model1)

    model_dict = model1.state_dict()
    pretrained_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    
    # print(pretrained_dict)
    model_dict.update(pretrained_dict)
    model1.load_state_dict(model_dict)

    validate(val_loader, model1)

    return

def exp5():
    
    print('l**********')
    for j1 in range(len(lst)):
        l1 = lst[j1][:,1]
        x1 = np.average(Variable(l1).data.numpy())
        # y1 = ylabel[j1]
        print(str(x1))

    print('c**********')
    for j1 in range(len(lst)):
        l1 = lst[j1][:,2]
        x2 = np.average(Variable(l1).data.numpy())
        # y1 = ylabel[j1]
        print(str(x2))
    
    print('s**********')
    for j1 in range(len(lst)):
        l1 = lst[j1][:,3]
        x3 = np.average(Variable(l1).data.numpy())
        # y1 = ylabel[j1]
        print(str(x3))

    print('ef**********')
    for j1 in range(len(lst)):
        l1 = lst[j1][:,7]
        x4 = np.average(Variable(l1).data.numpy())
        # y1 = ylabel[j1]
        print(str(x4))

    print('no**********')
    for j1 in range(len(lst)):
        l1 = lst[j1][:,8]
        x5 = np.average(Variable(l1).data.numpy())
        # y1 = ylabel[j1]
        print(str(x5))


    print('layername**********')
    for j3 in range(len(lst)):
        y = layerLst[j3]
        print(str(y))

    
    fig = plt.figure()
    plt.plot(layerLst,[np.average(Variable(l1[:,1]).data.numpy()) for l1 in lst])
    plt.plot(layerLst,[np.average(Variable(l1[:,2]).data.numpy()) for l1 in lst])
    plt.plot(layerLst,[np.average(Variable(l1[:,3]).data.numpy()) for l1 in lst])
    plt.legend()
    plt.style.use('ggplot')
    plt.savefig(dir+'lcs.png')




@torch.no_grad()        # deactivate autograd engine to reduce memory consumption and speed up computations
def validate(val_loader, model):
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    for i, (id, img_name, input, target) in enumerate(val_loader):
        input = input.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)
        # compute output
        output = model(input)
        acc1, acc5 = accuracy(output, target, topk=(1, 1))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def load_checkpoint(checkpoint, model, optimizer=None):
    
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location=device)
    else:
        # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict']) 


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count






if __name__ == '__main__':
    exp1()
    # exp2()
    # exp3()
    # exp4()
    # exp5()

