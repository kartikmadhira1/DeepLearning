import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os 
import time 

#Parameters set:
cropSize=128
#range for perturbing the corners to get the homographies [-rho,+rho]
rho=16
#resize shape
resize=(320,240)
#number of train sets to be made
numTrainData=82780
#file path
filePath="/home/kartikmadhira/datasets/ms-coco/train2014/"
saveDest="/home/kartikmadhira/datasets/ms-coco/"

def loadTrainList(filePath,numTrainData):
    fileList=os.listdir(filePath)
    dataList=[]
    #take any random number and take 5000 of these images
    for i in range(numTrainData):
        dataList.append(filePath+fileList[i])
    return dataList


def saveData(savePath,data,i):
	if not os.path.exists(savePath+'data'):
	    os.makedirs(savePath+'data')
	if not os.path.exists(savePath+'labels'):
	    os.makedirs(savePath+'labels')
	np.savez(savePath+'data/'+str(i)+'.npz',data[0])
	np.savez(savePath+'labels/'+str(i)+'.npz',data[1])


def getImages(dataList,numTrainData,saveDest):
    #load image
    for i in range(numTrainData):
        
        image=cv2.imread(dataList[i],cv2.IMREAD_GRAYSCALE)
        image=cv2.resize(image,resize)
        #get a random x and y location that does not have the borders
        #x is Y and y is X!
        getLocX=random.randint(105,160)
        getLocY=random.randint(105,225)
        #crop the image
        patchA=image[getLocX-int(cropSize/2):getLocX+int(cropSize/2),getLocY-int(cropSize/2):getLocY+int(cropSize/2)]

        #perturb image randomly and apply homography
        pts1=np.float32([[getLocY-cropSize/2+random.randint(-rho,rho),getLocX-cropSize/2+random.randint(-rho,rho)],
              [getLocY+cropSize/2+random.randint(-rho,rho),getLocX-cropSize/2+random.randint(-rho,rho)],
              [getLocY+cropSize/2+random.randint(-rho,rho),getLocX+cropSize/2+random.randint(-rho,rho)],
              [getLocY-cropSize/2+random.randint(-rho,rho),getLocX+cropSize/2+random.randint(-rho,rho)]])
        pts2=np.float32([[getLocY-cropSize/2,getLocX-cropSize/2],
              [getLocY+cropSize/2,getLocX-cropSize/2],
              [getLocY+cropSize/2,getLocX+cropSize/2],
              [getLocY-cropSize/2,getLocX+cropSize/2]])

        #get the perspective transform
        hAB=cv2.getPerspectiveTransform(pts2,pts1)
        #get the inverses
        hBA=np.linalg.inv(hAB)
        #get the warped image from the inverse homography generated in the dataset
        warped=np.asarray(cv2.warpPerspective(image,hBA,resize)).astype(np.uint8)
        #get the last patchB at the same location but on the warped image.
        patchB=warped[getLocX-int(cropSize/2):getLocX+int(cropSize/2),getLocY-int(cropSize/2):getLocY+int(cropSize/2)]
        #stack images on top of each other.
        stackedData=np.dstack((patchA,patchB))
        # #homogrpahy check
        # orig=cv2.warpPerspective(patchB,hAB,(128,128))
        # plt.subplot(1,2,1)
        # plt.imshow(patchA)
        # plt.subplot(1,2,2)
        # plt.imshow(patchB)
        if(i%3000==0):
            print('Saved '+str(i)+' images')
        saveData(saveDest,[stackedData,hAB],i)

#get the files

files=loadTrainList(filePath,numTrainData)
getImages(files,numTrainData,saveDest)
