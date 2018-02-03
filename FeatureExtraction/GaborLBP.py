# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 12:37:05 2018

@author: IssaMawad
"""
from skimage import data,exposure
from skimage.filters import gabor_kernel
from PIL import Image
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct,idct
#from zernike import Zernikemoment

import skimage.transform as trans
def extractGabor(img):
    """extracts Gabor features using a 8 orientations,5 frequencies Gabor bank

    Args:
        img: the image to be analysed 
    Returns:
        a feature vector that represents the concatenated the real parts of the 2D convolution 
        between the input image and each filter of the Gabor bank after resizing each result to 32x32
        the dimention is 40960(40*32*32)
    
    """
    block_overlap = 0 # one or two parameters for block overlap
          # Gabor parameters
    gabor_directions = 8
    gabor_scales = 5
    gabor_sigma = 2. * np.pi
    gabor_maximum_frequency = np.pi / 2.
    gabor_frequency_step = np.sqrt(.5)
    gabor_power_of_k = 0
    gabor_dc_free = True
    use_gabor_phases = False
      # LBP parameter
    lbp_radius = 2
    lbp_neighbor_count = 8
    lbp_uniform = True
    lbp_circular = True
    lbp_rotation_invariant= False
    lbp_compare_to_average = False
    lbp_add_average = False
      # histogram options
    sparse_histogram = False
    split_histogram = None
    block_size = 10
    block_overlap = 4
    #output = np.zeros(400)
    output = np.zeros(1024*40)
    i = 0;
    for theta in range(8):
        theta = theta / 8. * np.pi
        for frequency in ((np.pi/2) - (4*gabor_frequency_step),(np.pi/2) - (3*gabor_frequency_step),(np.pi/2) - (2*gabor_frequency_step),(np.pi/2) - gabor_frequency_step,np.pi/2):
            kernel = gabor_kernel(frequency, theta=theta,sigma_x=gabor_sigma,sigma_y=gabor_sigma)
            filtered = np.abs(ndi.convolve(img, kernel, mode='wrap'))
                
            #lbp = local_binary_pattern(filtered, lbp_neighbor_count, lbp_radius, 'uniform')
            #n_bins = int(lbp.max() + 1)
            #h = np.histogram(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins))
            #output[i:i+10] = h[0]
            output[i:i+1024]= trans.resize(filtered,[32,32]).ravel()
            i = i+1024
    return output

            

  
      
    
        
     
        