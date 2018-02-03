# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 16:29:12 2018

@author: IssaMawad
"""
from skimage import data,exposure
import scipy.misc
import dlib
import os;
def align(inputpath,outputpath,imgExt='pgm'):
   detector = dlib.get_frontal_face_detector()
   if not os.path.exists(outputpath):
        os.makedirs(outputpath)
   for subdir, dirs, files in os.walk(inputpath):
       for dir in dirs:
        dirFull = os.path.join(inputpath,dir)
        dirToMake = os.path.join(outputpath,dir)
        print(dirToMake)
        if not os.path.exists(dirToMake):
            os.makedirs(dirToMake)
        index=0
        for innerSubDir,innerDirs,innerFiles in os.walk(dirFull):
            
            print(len(innerFiles))
            if(len(innerFiles)==1):
                continue;
            for file in innerFiles:
                if(not file.endswith(imgExt)):
                    continue;
                fullFile = os.path.join(dirFull,file)
                img = data.imread(fullFile)
                dets = detector(img,1)
                for i, d in enumerate(dets):
                    
                    
                    x = img[d.top():d.bottom(),d.left():d.right()]
                    imgNewPath = dirToMake+'\\'+str(index)+'.jpg'
                    #print(imgNewPath)
                    if not os.path.exists(imgNewPath):
                        try:
                            scipy.misc.imsave(imgNewPath, x)
                        except:
                            continue
                    index = index+1
                
