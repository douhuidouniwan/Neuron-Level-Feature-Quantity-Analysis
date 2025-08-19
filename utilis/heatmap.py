import cv2
import numpy as np
def heatmap_general(img_ori:np.ndarray,img_mask:np.ndarray)->np.ndarray :
    #gray to colormap
    heatmap = cv2.applyColorMap(img_mask,cv2.COLORMAP_JET)
    #blur
    heatmap = cv2.GaussianBlur(heatmap,(3,3),sigmaX=1)
    #
    result = cv2.addWeighted(img_ori,0.5,heatmap,0.5,0)
    return result