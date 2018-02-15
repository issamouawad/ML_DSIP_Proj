"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import cnn.facenet

import os

import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate,misc
from os import listdir
from os.path import isfile, join



def EvaluateCNN(path,model_path):
    """
    Feeds the images stored in a specified path to the FaceNet CNN using the specified weights
  
    arguments: 
        path: the directory to look in, it is expected the each class of faces has a sub-directory, and images are resized to 160x160
        model_path: the path of the parameters model to configure the network with
        returns:
            x: nx128 embeddings extracted from n images
            y: the labels (classes) of images, the sub-directory name is used as a class name
    """
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            
            curDir = os.path.dirname(__file__)
            paths = []
            classes = []
            for subdir, dirs, files in os.walk(path):
                for dir in dirs:
                    dirFull = os.path.join( path,dir)
                    for innerSubDir,innerDirs,innerFiles in os.walk(dirFull):
                        for file in innerFiles:
                            classes.append(dir)
                            fullFile = os.path.join(dirFull,file)
                            paths.append(fullFile)
                            
            
            classes = np.asarray(classes)
            cnn.facenet.load_model(model_path)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            
            #image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
            image_size = 160
            
            embedding_size = embeddings.get_shape()[1]
           
            
            
            images = cnn.facenet.load_data(paths, False, False, image_size)
            feed_dict = { images_placeholder:images, phase_train_placeholder:False }
            
            # Run forward pass to calculate embeddings
            print('Runnning forward pass on LFW images')
            batch_size =100
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
        
            for i in range(nrof_batches):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = cnn.facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
               
            
            return (emb_array,classes)
            
            
