#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:05:35 2019

@author: kartikmadhira
"""

import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os 
import time 
import tensorflow as tf
from Network.Network import HomographyModel
import argparse 


def loadTrainList(filePath,numTrainData,numImagesLimit):
    fileList=os.listdir(filePath)
    dataList=[]
    #take any random number and take 5000 of these images
    for i in range(numTrainData):
        randInt=random.randint(1,numImagesLimit-1)
        dataList.append(filePath+fileList[randInt])
    return dataList


def saveData(savePath,data,i):
	if not os.path.exists(savePath+'data'):
	    os.makedirs(savePath+'data')
	if not os.path.exists(savePath+'labels'):
	    os.makedirs(savePath+'labels')
	np.savez(savePath+'data/'+str(i)+'.npz',data[0])
	np.savez(savePath+'labels/'+str(i)+'.npz',data[1])


def getImages(firstImage,secondImage,saveDest,ModelPath):
    #load image
    cropSize=128
    resize=(320,240)
    rho=22
    firstI=cv2.imread(firstImage)
    
    image1=cv2.imread(firstImage,cv2.IMREAD_GRAYSCALE)
    image=cv2.resize(image1,resize)
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
        #get the inverse
    hBA=np.linalg.inv(hAB)
        #get the warped image from the inverse homography generated in the dataset
    warped=np.asarray(cv2.warpPerspective(image,hAB,resize)).astype(np.uint8)
        #get the last patchB at the same location but on the warped image.
    patchB=warped[getLocX-int(cropSize/2):getLocX+int(cropSize/2),getLocY-int(cropSize/2):getLocY+int(cropSize/2)]
    
    
    cv2.line(firstI,   (pts1[0][0],pts1[0][1]),(pts1[1][0],pts1[1][1]), (255,0,0), 3)
    cv2.line(firstI,  (pts1[1][0],pts1[1][1]),(pts1[2][0],pts1[2][1]), (255,0,0), 3)
    cv2.line(firstI,  (pts1[2][0],pts1[2][1]),(pts1[3][0],pts1[3][1]), (255,0,0), 3)
    cv2.line(firstI,  (pts1[3][0],pts1[3][1]),(pts1[0][0],pts1[0][1]), (255,0,0), 3)

    ImageSize = [128, 128, 2]
    ImgPH=tf.placeholder(tf.float32, shape=(1, 128, 128, 2))
    
    
    H4pt = HomographyModel(ImgPH, ImageSize, 1)
    Saver = tf.train.Saver()
  
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        
        Img=np.dstack((patchA,patchB))
        image=Img
        Img=np.array(Img).reshape(1,128,128,2)
        
        FeedDict = {ImgPH: Img}
        PredT = sess.run(H4pt,FeedDict)
        #label=label.reshape(1,8)
        #print(PredT,label)
        #loss=np.sqrt(np.mean((PredT-label)**2))
        #print(loss)
        plt.subplot(2,1,1)
        plt.imshow(image[:,:,0])
        plt.subplot(2,1,2)
        plt.imshow(image[:,:,1])
    
    newPointsDiff=PredT.reshape(4,2)
    print(newPointsDiff)
    pts2=np.float32([[getLocY-cropSize/2,getLocX-cropSize/2],
          [getLocY+cropSize/2,getLocX-cropSize/2],
          [getLocY+cropSize/2,getLocX+cropSize/2],
          [getLocY-cropSize/2,getLocX+cropSize/2]])
    pts1=pts2+newPointsDiff
    H4pts=pts2-pts1
    #get the perspective transform
    hAB=cv2.getPerspectiveTransform(pts2,pts1)
    #get the inverse
    hBA=np.linalg.inv(hAB)
    #get the warped image from the inverse homography generated in the dataset
    #warped=np.asarray(cv2.warpPerspective(firstImageCol,hAB,)).astype(np.uint8)
    #get the last patchB at the same location but on the warped image.
    #patchB=warped[getLocX-int(cropSize/2):getLocX+int(cropSize/2),getLocY-int(cropSize/2):getLocY+int(cropSize/2)]

    cv2.line(firstI,   (pts1[0][0],pts1[0][1]),(pts1[1][0],pts1[1][1]), (0,0,255), 3)
    cv2.line(firstI,  (pts1[1][0],pts1[1][1]),(pts1[2][0],pts1[2][1]),(0,0,255), 3)
    cv2.line(firstI,  (pts1[2][0],pts1[2][1]),(pts1[3][0],pts1[3][1]),(0,0,255), 3)
    cv2.line(firstI,  (pts1[3][0],pts1[3][1]),(pts1[0][0],pts1[0][1]), (0,0,255), 3)
    #plt.figure()
    #plt.imshow(warped)
    #plt.show()
    cv2.imwrite('result'+'.png',firstI)

    #stack images on top of each other.
    #stackedData=np.dstack((patchA,patchB))
    # #homogrpahy check
    # orig=cv2.warpPerspective(patchB,hAB,(128,128))
    # plt.subplot(1,2,1)
    # plt.imshow(patchA)
    # plt.subplot(1,2,2)
    # plt.imshow(patchB)

def main():
    rand=random.randint(0,100)
    
    Parser = argparse.ArgumentParser()
    
    Parser.add_argument('--Image', default='/home/kartikmadhira/CMSC733/YourDirectoryID_p1/Phase2/Data/Val/10.jpg', help='Images')
    Parser.add_argument('--ModelPath', default='/home/kartikmadhira/CMSC733/YourDirectoryID_p1/Phase2/Checkpoints/169model.ckpt', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--ModelType', default='Sup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    
    
    Args = Parser.parse_args()
    Image = Args.Image
    ModelPath = Args.ModelPath
    ModelType = Args.ModelType
    #first='/home/kartikmadhira/CMSC733/YourDirectoryID_p1/Phase2/Data/Val/'+str(rand)+'.jpg'
    getImages(Image,rand,'new',ModelPath)

     
if __name__ == '__main__':
    main()

    
