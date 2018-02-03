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

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import neighbors
from sklearn.model_selection import train_test_split

def ClassifyKCrossValidation(x,y,num_splits=2,Classifier='svm',kernel='linear',poly_degree=3,params=[0.5,1,1.5],distance_metric='euclidean',use_dimentionality_reduction='false',dimentionality_reduction='lda',pca_components = 50):
    """this function performs classification based on data matrix x and labels y.
    data (and labels) are split into train and test sets, then train set is split into train and validation sets based on k-fold leave out cross validation in order to choose the model parameters that yield the least classification error
    finally the model (with the chosen parameter) is tested with the test data
    Args:
        x: The data matrix n_samples rows X n_variables columns
        y: labels of the data n_samples label
        num_splits: number of splits used for performing k-fold holdout cross validation
        
        Classifier: the type of classifier model , default is svm.
            possible values are svm, knn
        kernel: kernel type of the classifier default is linear
            possible values are linear, poly,rbf,sigmoid (only applicable with kernalized models)
        poly_degree: degree of polynomial when used as a kernel 
            default is 3
        params: array of parameters of the classifier model
            integers in case of knn representing parameter k , (usually an odd number)
            real numbers in case of other methods
        distance_metric: distance metric to use in knn classifier , default is euclidean
            possible values are manhattan,euclidean
        use_dimentionality_reduction: 'true' or 'false' 
        dimentionality_reduction: the method to use for dimensionality reduction , default is 'pca'
        possible values are: 'pca','lda','both' (in both , first pca is computed then lda)
        pca_components: integer determines the number of pca components to use for projection the data to 
    
    Returns:
        prints the classification report
    
    """
    skf = StratifiedKFold(n_splits=num_splits)
   
    X_train, X_test, y_train, y_test = train_test_split(x,y,stratify=y)
    
    globalScore = np.zeros(len(params))
    score_max=0
    j_max=-1
    
    singleRunScore = np.zeros(num_splits)
    target_names = np.unique(y)
    
    
    for j in range(len(params)):
        split = 0
        for train, val in skf.split(X_train, y_train):
            
            xtr = X_train[train]
            ytr = y_train[train]
            xval = X_train[val]
            yval = y_train[val]
            out = ClassifySimple(xtr,ytr,xval,params[j],Classifier,kernel,poly_degree,distance_metric,use_dimentionality_reduction,dimentionality_reduction,pca_components)
            
            p =precision_score(yval,out,average='micro')
            
            print(classification_report(yval, out, target_names=target_names))
            singleRunScore[split] = p
            split = split+1
        
        globalScore[j] = np.mean(singleRunScore)
        print('mean error for parameter '+str(j)+' is '+ str(globalScore[j]))
        if(globalScore[j]>=score_max):
            score_max = globalScore[j]
            j_max = j
    print('using test data:')
    print(j_max)
    
    test_out = ClassifySimple(X_train,y_train,X_test,params[j_max],Classifier,kernel,poly_degree,distance_metric,use_dimentionality_reduction,dimentionality_reduction,pca_components )
    print(classification_report(y_test, test_out, target_names=target_names))             
    plt.plot(params,globalScore)
    
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
    model.fit(xtr,ytr)
    
    return model.predict(xts)