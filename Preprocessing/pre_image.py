# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 20:26:37 2018

@author: IssaMawad
"""
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data,exposure,color
import skimage.transform as trans
def ReadImage(fileName):
    img = data.imread(fileName);
    if(len(img.shape)>2):
        imgGray = color.rgb2gray(img)
    else:
        imgGray = img
    if(imgGray.shape[0]>64 or imgGray.shape[1]>64):
        imgGray = trans.resize(imgGray,[64,64],mode='reflect')
    imgGray = exposure.equalize_adapthist(imgGray)
    return imgGray