# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 18:43:22 2018

@author: IssaMawad
"""
import numpy as np
class RLSClassifier:
    """
    This Class Represents a Regularized-Kernelized Least square multi-class classifier
    based on single vs all classification to support multi-class classifications , and regularized by lambda
    
    """
    def __init__(self, param, kernel='linear', kernel_parameter=1):
        """
        declares a new object of the class constructed with 
        param: the regularization parameter lambda 
        kernel: the kernel type (default: linear) possible values are:
            gaussian, poly
        kernel_parameter: the parameter to use in case of kernelized classification (default value is 1), possible interpretations are:
            if kernel='poly' -> degree of the polynomial
            if kernel='gaussian'-> variance of the gaussian
        """
        self.param = param
        self.kernel = kernel
        
        
    def fit(self, x,y):
        """
        Fits the model using an input set (x) and output labels y
        this method doesn't return any value, it just stores the weights internally
        """
        target_names = np.unique(y)
        self.target_names = target_names
        reshapedY = -np.ones((x.shape[0],target_names.shape[0]))
        n= x.shape[0]
        d = x.shape[1]
        for i in range(target_names.shape[0]):
            reshapedY[np.where(y==target_names[i]),i]=1
            
        diagonal = np.dot(np.transpose(x),x)
        toInvert = diagonal + self.param*n*np.eye(d,d);
        inverted = np.linalg.inv(toInvert)
        self.w = np.dot(np.dot(inverted,np.transpose(x)),reshapedY)
            
    
    def predict(self,x):
        """
        evaluates a test data x and predicts the labels of each inputby applying the weights already stored.
        you must call this function only when you've already called fit.
        in case of kernelized classification, the training data are stored and used to compute the inner products with the test data
        """
        mul = np.dot(x,self.w)
        indices = np.argmax(mul,1)
        
        return self.target_names[indices]