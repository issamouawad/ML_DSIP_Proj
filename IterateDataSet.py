# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 13:09:23 2018

@author: IssaMawad
"""
import os;
from FeatureExtraction.HOGFeatures import extractHOGFeatures
from FeatureExtraction.GaborDown import extractGabor
#from FeatureExtraction.GaborZernike import extractGaborZernike
from FeatureExtraction.GaborDCT import extractGaborDCT
from FeatureExtraction.LGBPHS import extractLGBPHS
from cnn.Evaluation import EvaluateCNN
import numpy as np
from Preprocessing.pre_image import ReadImage 
def ExtractDataSetFeatures(dir_name,imageExt ='pgm',features_type='hog',cnn_model_path='C:\\Users\\IssaMawad\\Documents\\DSIP_ML_Project\\cnn\\models\\20170512-110547.pb'):
   """Extract features of image dataset using a defined method
       images will be converted to gray-scale (if not already) and the illumination will be normalized and resized to 64x64
    Args:
        dir_name: the main folder to process, it's expected that folder is organized into sub-folders, each of which represents a class
        imageExt: the image extension, can be any valid extension for images
            (default is pgm)
        features_type: the method to use in order to extract features. Possible values are:
            hog: specifying HOG features to be extracted(histogram of oriented gradients)
            gbr: specifying the use of Gabor filter bank of 40 Gabor filters
            plain: the image pixels are flattened (after pre-processing) in order to support eign-faces and fisher-faces 
            facenet: feeds the images (after resizing them to 160x160) to a pre-trained convolutional neural network (facenet) and the features are the learned embeddings.
        cnn_model_path: the absolute path of the model file of the facenet cnn, it's optional because it's not used with other feature types, but it's mandatory with facenet 
        
            
            
    Returns:
        x: data matrix n_samples rows (represents the number of images found in the directory)x n_features columns (depends on the method used)
        y: labels matrix, the sub-directory names are used as labels for the contained images
    
    """
   if(features_type=='facenet'):
      return EvaluateCNN(dir_name,cnn_model_path)
   features = [];
   classes = [];
   for subdir, dirs, files in os.walk(dir_name):
       for dir in dirs:
        dirFull = os.path.join( dir_name,dir)
        for innerSubDir,innerDirs,innerFiles in os.walk(dirFull):
            #if(len(innerFiles)<8):
            #    continue;
            #print(dir_name)
            for file in innerFiles:
                if(not file.endswith(imageExt)):
                    continue;
                fullFile = os.path.join(dirFull,file)
                if(features_type=='hog'):
                    features.append(extractHOGFeatures(ReadImage(fullFile)))
                if(features_type=='gbrdct'):
                    features.append(extractGaborDCT(ReadImage(fullFile)))
                if(features_type=='lgbphs'):
                    features.append(extractLGBPHS(ReadImage(fullFile)))
                if(features_type=='gabor'):
                    features.append(extractGabor(ReadImage(fullFile)))
                #if(features_type=='gbrzk'):
                  #  features.append(extractGaborZernike(ReadImage(fullFile)))
                if(features_type=='dum'):
                    features.append(ReadImage(fullFile).ravel())
                classes.append(dir)
   return np.asarray(features),np.asanyarray(classes)

    
    