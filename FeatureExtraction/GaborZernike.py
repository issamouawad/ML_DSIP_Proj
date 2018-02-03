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
from zernike import Zernikemoment
import skimage.transform as trans
def extractGaborZernike(fileName):
    img = data.imread(fileName);
    img = exposure.equalize_adapthist(img)
    img= trans.resize(img,[64,64])
       # one or two parameters for block size
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
    output = np.zeros(200)
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
            _v,output[i],_s= Zernikemoment(filtered,5,1)
            i = i+1
            _v,output[i],_s= Zernikemoment(filtered,6,0)
            i = i+1
            _v,output[i],_s= Zernikemoment(filtered,7,2)
            i = i+1
            _v,output[i],_s= Zernikemoment(filtered,5,1)
            i = i+1
            _v,output[i],_s= Zernikemoment(filtered,8,5)
            i = i+1
            
    return output

            

  
      
    
        
     
        