# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 12:15:35 2018

@author: IssaMawad
"""

from sklearn.model_selection import StratifiedKFold
from sklearn import svm
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from Classification.RLSClassifier import RLSClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from scipy.stats import chisquare

def ClassifyKCrossValidation(x,y,train_size=0.75,test_size=0.25,num_splits=2,Classifier='svm',kernel='linear',poly_degree=2,params=[0.5,1,1.5],distance_metric='euclidean',use_dimentionality_reduction='false',dimentionality_reduction='lda',pca_components = 50):
    """this function performs classification based on data matrix x and labels y.
    data (and labels) are split into train and test sets, then train set is split into train and validation sets based on k-fold leave out cross validation in order to choose the model parameters that yield the least classification error
    finally the model (with the chosen parameter) is tested with the test data
    Args:
        x: The data matrix n_samples rows X n_variables columns
        y: labels of the data n_samples label
        num_splits: number of splits used for performing k-fold cross validation
        
        Classifier: the type of classifier model , default is svm.
            possible values are svm, knn,rls
        kernel: kernel type of the classifier default is linear
            possible values are     
                knn: N/A
                rls: linear, gaussian, poly
                svm, linear, rbf, poly, sigmoid
        kernel_param: additional parameter used for some kernels(default is 2)
            poly: polynomial degree
            gaussian: sigma
        params: array of parameters of the classifier model
            integers in case of knn representing parameter k , (usually an odd number)
            real numbers in case of other methods representing the regularizer
        distance_metric: distance metric to use in knn classifier , default is euclidean
            possible values are manhattan,euclidean
        use_dimentionality_reduction: 'true' or 'false' 
        dimentionality_reduction: the method to use for dimensionality reduction , default is 'pca'
        possible values are: 'pca','lda','both' (in both , first pca is computed then lda)
        pca_components: integer determines the number of pca components to use for projection the data to 
    
    Returns:
        prints the classification report
    
    """
    
   
    X_train, X_test, y_train, y_test = train_test_split(x,y,stratify=y,train_size=train_size, test_size=test_size)
    
    globalScore = np.zeros(len(params))
    score_max=0
    j_max=-1
    
    singleRunScore = np.zeros(num_splits)
    target_names = np.unique(y)
    
    skf = StratifiedKFold(n_splits=num_splits)
    
    for j in range(len(params)):
        split = 0
        for train, val in skf.split(X_train, y_train):
            
            xtr = X_train[train]
            ytr = y_train[train]
            xval = X_train[val]
            yval = y_train[val]
            out = ClassifySimple(xtr,ytr,xval,params[j],Classifier,kernel,poly_degree,distance_metric,use_dimentionality_reduction,dimentionality_reduction,pca_components)
            
            p =precision_score(yval,out,average='micro')
            
            print('using parameter = '+str(params[j])+', fold  accuracy is ' + str(p))
            
            singleRunScore[split] = p
            split = split+1
        
        globalScore[j] = np.mean(singleRunScore)
        print('mean accuracy for parameter '+str(params[j])+' is '+ str(globalScore[j]))
        if(globalScore[j]>=score_max):
            score_max = globalScore[j]
            j_max = j
    fig = plt.figure()
    plt.plot(params,globalScore)
    fig.suptitle('Average Accuracy for Each Parameter')
    plt.xlabel('Parameter Value')
    plt.ylabel('Accuracy %')
    print('training the model on all the training set using parameter=' + str(params[j_max]))
    
    
    test_out = ClassifySimple(X_train,y_train,X_test,params[j_max],Classifier,kernel,poly_degree,distance_metric,use_dimentionality_reduction,dimentionality_reduction,pca_components )
    print(classification_report(y_test, test_out, target_names=target_names))             
    
    
def ClassifySimple(xtr,ytr,xts,param,Classifier='svm',kernel='linear',poly_degree=3,distance_metric='euclidean',use_dimentionality_reduction='false',dimentionality_reduction='lda',components = 50):
    if(use_dimentionality_reduction=='true'):
                if(dimentionality_reduction == 'lda'):
                    lda = LinearDiscriminantAnalysis(n_components=components).fit(xtr,ytr)
                    xtr = lda.transform(xtr)
                    xts = lda.transform(xts)
                else:
                    if(dimentionality_reduction == 'pca'):
                        pca = PCA(components).fit(xtr)
                        xtr = pca.transform(xtr)
                        xts = pca.transform(xts)
                    else:
                        if(dimentionality_reduction == 'both'):
                            pca = PCA(components).fit(xtr)
                            xtr = pca.transform(xtr)
                            xts = pca.transform(xts)
                            lda = LinearDiscriminantAnalysis(n_components=components).fit(xtr,ytr)
                            xtr = lda.transform(xtr)
                            xts = lda.transform(xts)
                            
    if(Classifier=='svm'):
        model = svm.SVC(kernel=kernel,degree=3,C=param)
    if(Classifier=='knn'):
        model = neighbors.KNeighborsClassifier(param,metric=distance_metric)
    if(Classifier =='RLS'):
        model = RLSClassifier(param,kernel)
        #model = neighbors.KNeighborsClassifier(param, metric='pyfunc', metric_params={"func":histogram_intersection})
    model.fit(xtr,ytr)
    
    return model.predict(xts)
def chisquare_distance(a,b):
    c,p =  chisquare(a,b)
    print(p)
    return c
def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    dist= np.sum(p[filt] * np.log2(p[filt] / q[filt]))
    return dist
def histogram_intersection(h1, h2):
   
    sm = 0
    for i in range(len(h1)):
        sm += min(h1[i],h2[i])
    return 1-sm

        