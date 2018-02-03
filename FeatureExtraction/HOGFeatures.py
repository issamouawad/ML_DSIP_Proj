# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 12:44:17 2018

@author: IssaMawad
"""
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data,exposure,color
import skimage.transform as trans


def extractHOGFeatures(img):
    """extracts HOG features using the optimal parameters acquired after many tests
    
        Args:
        img: the image to be analysed 
        Returns:
        a feature vector that represents the histogram of oriented gradients with the parameters specified.
        
    """    
    #print(img.shape)
    f = hog(img,block_norm='L2-Hys',pixels_per_cell=[6,6],cells_per_block=[4,4],orientations=5)
    #f = hog(img)
    return f
 
    