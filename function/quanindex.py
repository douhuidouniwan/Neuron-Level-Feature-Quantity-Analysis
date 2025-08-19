import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import sys
import gc



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size=11, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def fq(img1, img2, window_size=11, channel=1, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
        window = window.type_as(img1)


    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2
    sigma1 = sigma1_sq.sqrt()
    sigma2 = sigma2_sq.sqrt()
    sigma1_sigma2 = sigma1*sigma2

    C1 = 0.01**2
    C2 = 0.03**2
    C3 = C2/2.0
    
    arr = torch.zeros(7)


    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    l = (2*mu1_mu2)/(mu1_sq + mu2_sq + C1)  #Luminance Comparison
    c = (2*sigma1_sigma2)/(sigma1_sq + sigma2_sq + C2) #Contrast Comparison
    s = (sigma12)/(sigma1_sigma2 +C3) #Structure Comparison
    # sum_map1 = ((2*mu1_mu2)*(2*sigma12))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    sum_map = ((2*mu1_mu2)*(2*sigma1_sigma2)*(sigma12))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)*(sigma1_sigma2 +C3))

    if size_average:

        arr[0] = ssim_map.mean()
        arr[1] = l.mean()
        arr[2] = c.mean()
        arr[3] = s.mean()
        arr[4] = mu2.mean()
        arr[5] = sigma2.mean()
        arr[6] = sum_map.mean()
    else:
        arr[0] = ssim_map.mean(1).mean(1).mean(1)
        arr[1] = l.mean(1).mean(1).mean(1)
        arr[2] = c.mean(1).mean(1).mean(1)
        arr[3] = s.mean(1).mean(1).mean(1)
    
    return arr


def quantity(efsc,nosc,foreimage,n_img):
    arr = torch.zeros(2)
    n_ef = torch.nonzero(efsc).size(0)
    n_no = torch.nonzero(nosc).size(0)
    n_fo = torch.nonzero(foreimage).size(0)
    eff = (n_ef*n_img)/(n_fo*n_no+0.0001)
    arr[0] = n_ef/(n_fo+0.00001)
    arr[1] = n_no/(n_img+0.00001)
    return arr


def mask(image):
    threshold_value = 0
    maskimage = (image > threshold_value).float()
    return maskimage


def quanindex(foreimage, sc, n_img):

    foremask = mask(foreimage)
    scmask = mask(sc)
    efsc = foremask * sc
    nosc = sc - efsc

    efimage = scmask * foreimage

    index1 = fq(efimage, efsc)

    index2 = quantity(efsc,nosc,foreimage,n_img)

    index = torch.cat([index1,index2],0).detach().clone()

    return index
