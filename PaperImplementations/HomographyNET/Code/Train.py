#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:00:39 2019
@author: kartikmadhira
"""
import tensorflow as tf
import cv2
import sys
import glob
import math
#import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.NetworkUnsupervised import  Unsupervised_HomographyModel
from Network.Network import  HomographyModel
#from Misc.MiscUtils import *
#from Misc.DataUtils import *
import numpy as np
import argparse
from StringIO import StringIO
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from utils.gatherDataSupervised import generateImagesSupervised
from utils.gatherDataUnsupervised import generateImagesUnsupervised
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *

# Don't generate pyc codes
sys.dont_write_bytecode = True


def GenerateBatchSupervised(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize,NumTrainImages):
    
    #get the basepath to the folder data/ where both train images and labels are present
    
    
    """
    Inputs: 
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    I1Batch = []
    LabelBatch = []
    
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, NumTrainImages-1)
        
        RandImagename=BasePath+'/data/'+str(RandIdx)+'.npz'        
        ImageNum += 1
    	npzfile=np.load(RandImagename)
        image=npzfile['arr_0']
 
        I1 = np.float32(image)
        I1=(I1-np.mean(I1))/255
        Label = BasePath+'/labels/'+str(RandIdx)+'.npz' 
        npzfile=np.load(Label)
        labelRegress=npzfile['arr_0']
        labelRegress.resize((8,1))
        labelRegress=labelRegress[:,0]
        # Append All Images and Mask
        I1Batch.append(I1)
        LabelBatch.append(labelRegress)
        
    return I1Batch, LabelBatch

def GenerateBatchUnsupervised(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize,NumTrainImages):
    
    #get the basepath to the folder data/ where both train images and labels are present
    
    
    """
    Inputs: 
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    stackedDataBatch=[]
    IABatch = []
    cornerBatch = []
    
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, NumTrainImages-1)
        
        RandImagename=BasePath+'/unsupervised/data/'+str(RandIdx)+'.npz'        
        ImageNum += 1
    	npzfile=np.load(RandImagename)
        image=npzfile['arr_0']
    	##########################################################
    	# Add any standardization or data augmentation here!
    	##########################################################
        I1 = np.float32(image)
        I1=(I1-np.mean(I1))/255
        #print(I1.shape)
        
        RandIAname=BasePath+'/unsupervised/Ia/'+str(RandIdx)+'.npz'        
    	npzfile=np.load(RandIAname)
        image=npzfile['arr_0']
        Ia = np.float32(image)
        Ia=(Ia-np.mean(Ia))/255
              
        
        Label = BasePath+'/unsupervised/cornerData/'+str(RandIdx)+'.npz' 
        npzfile=np.load(Label)
        labelRegress=npzfile['arr_0']
        # Append All Images and Mask
        labelRegress.resize((8,1))
        labelRegress=labelRegress[:,0]
        stackedDataBatch.append(I1)
                
        IABatch.append(Ia)
        cornerBatch.append(labelRegress)

        
    return stackedDataBatch, IABatch , cornerBatch


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              


def TrainOperationUnsupervised(ImgPH,CornerPH,I2PH,DirNamesTrain, TrainLabels,NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, ModelType,NumTrainImages):
    
    pred_I2,I2 = Unsupervised_HomographyModel(ImgPH, CornerPH, I2PH,None, ImageSize, MiniBatchSize)

    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.abs(pred_I2 - I2))
    with tf.name_scope('Adam'):
        Optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

	# Tensorboard
	# Create a summary to monitor loss tensor
	EpochLossPH = tf.placeholder(tf.float32, shape=None)
	loss_summary = tf.summary.scalar('LossEveryIter', loss)
	epoch_loss_summary = tf.summary.scalar('LossPerEpoch', EpochLossPH)
	# tf.summary.image('Anything you want', AnyImg)
	# Merge all summaries into a single operation
	MergedSummaryOP1 = tf.summary.merge([loss_summary])
	MergedSummaryOP2 = tf.summary.merge([epoch_loss_summary])
	# MergedSummaryOP = tf.summary.merge_all()

	# Setup Saver
	Saver = tf.train.Saver()
	with tf.Session() as sess:       
	    if LatestFile is not None:
	        Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
	        # Extract only numbers from the name
	        StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
	        print('Loaded latest checkpoint with the name ' + LatestFile + '....')
	    else:
	        sess.run(tf.global_variables_initializer())
	        StartEpoch = 0
	        print('New model initialized....')

	    # Tensorboard
	    Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())
	       # Define PlaceHolder variables for Input and Predicted output
        #
        for Epochs in tqdm(range(StartEpoch, NumEpochs)):
	        NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
	        Loss=[]
	        epoch_loss=0
	        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
	            patchBatch, IABatch,cornerBatch= GenerateBatchUnsupervised(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize,NumTrainImages)
	            FeedDict = {ImgPH: patchBatch, CornerPH: cornerBatch, I2PH: IABatch}
	            _, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP1], feed_dict=FeedDict)
	            #print(shapeH4pt,shapeLabel).
	            Loss.append(LossThisBatch)
	            epoch_loss = epoch_loss + LossThisBatch
	            # Save checkpoint every some SaveCheckPoint's iterations
	            if PerEpochCounter % SaveCheckPoint == 0:
	                # Save the Model learnt in this epoch
	                SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
	                Saver.save(sess,  save_path=SaveName)
	                print('\n' + SaveName + ' Model Saved...')

	            # Tensorboard
	            Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
	          

	        epoch_loss = epoch_loss/NumIterationsPerEpoch
	        
	        print(np.mean(Loss))
	        # Save model every epoch
	        SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
	        Saver.save(sess, save_path=SaveName)
	        print('\n' + SaveName + ' Model Saved...')
	        Summary_epoch = sess.run(MergedSummaryOP2,feed_dict={EpochLossPH: epoch_loss})
	        Writer.add_summary(Summary_epoch,Epochs)
	        Writer.flush()

def TrainOperationSupervised(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, ModelType,NumTrainImages):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    LabelPH is the one-hot encoded label placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
	ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """      
    # Predict output with forward pass
    H4pt = HomographyModel(ImgPH, ImageSize, MiniBatchSize)

    with tf.name_scope('Loss'):
        ###############################################
        # Fill your loss function of choice here!
        ###############################################
        #LabelPH=tf.reshape(LabelPH,[MiniBatchSize,LabelPH.shape[1:4].num_elements()])
        shapeH4pt=tf.shape(H4pt)
        shapeLabel=tf.shape(LabelPH)
        loss = tf.sqrt(tf.reduce_sum((tf.squared_difference(H4pt,LabelPH))))
    with tf.name_scope('Adam'):
    	###############################################
    	# Fill your optimizer of choice here!
    	###############################################
        Optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    # Tensorboard
    # Create a summary to monitor loss tensor
    tf.summary.scalar('LossEveryIter', loss)
    # tf.summary.image('Anything you want', AnyImg)
    # Merge all summaries into a single operation
    MergedSummaryOP = tf.summary.merge_all()

    # Setup Saver
    Saver = tf.train.Saver()
    with tf.Session() as sess:       
        if LatestFile is not None:
            Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
            # Extract only numbers from the name
            StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
            print('Loaded latest checkpoint with the name ' + LatestFile + '....')
        else:
            sess.run(tf.global_variables_initializer())
            StartEpoch = 0
            print('New model initialized....')

        # Tensorboard
        Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())
            
        for Epochs in tqdm(range(StartEpoch, NumEpochs)):
            NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
            appendAcc=[]
            appendLoss=[]
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                I1Batch, LabelBatch = GenerateBatchSupervised(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize,NumTrainImages)
                FeedDict = {ImgPH: I1Batch, LabelPH: LabelBatch}
                _, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP], feed_dict=FeedDict)
                #print(shapeH4pt,shapeLabel).
                appendLoss.append(LossThisBatch)
                
                # Save checkpoint every some SaveCheckPoint's iterations
                if PerEpochCounter % SaveCheckPoint == 0:
                    # Save the Model learnt in this epoch
                    SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                    Saver.save(sess,  save_path=SaveName)
                    print('\n' + SaveName + ' Model Saved...')

                # Tensorboard
                Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                # If you don't flush the tensorboard doesn't update until a lot of iterations!
                Writer.flush()
            
            print(np.mean(appendLoss))
            # Save model every epoch
            SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
            Saver.save(sess, save_path=SaveName)
            print('\n' + SaveName + ' Model Saved...')




def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    
    Parser.add_argument('--BasePath', default='/home/kartikmadhira/CMSC733/YourDirectoryID_p1/Phase2/Data', help='Base path of images, Default:/media/nitin/Research/Homing/SpectralCompression/COCO')
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--ModelType', default='Unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Parser.add_argument('--NumTrainImages', type=int, default=127, help='Number of examples to train on from the train images set')
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=32, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    NumTrainImages=Args.NumTrainImages
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    if (ModelType=='Unsup'):
        # Define PlaceHolder variables for Input and Predicted output
        #tf.reset_default_graph()
        cropSize=128
        #range for perturbing the corners to get the homographies [-rho,+rho]
        rho=16
        #resize shape
        resize=(320,240)
        #number of train sets to be made
        numTrainData=128
        #limit of the number of the images in the train set
        numImagesLimit=5000
        #file path
        filePath=BasePath+'/Train/'
        if not os.path.exists(BasePath+''):
            os.makedirs(BasePath+'/unsupervised/')
        saveDest=BasePath+'/unsupervised/'
        
        generateImagesUnsupervised(cropSize,rho,resize,numTrainData,numImagesLimit,filePath,saveDest)
        
        print('===============Generated'+str(numTrainData)+'=========================')
        
        
        ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 128, 128, 2))
        CornerPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 8))
        I2PH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 128, 128,1))
        
        TrainOperationUnsupervised(ImgPH,CornerPH,I2PH,DirNamesTrain, TrainLabels,126, ImageSize,
                       NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                       DivTrain, LatestFile, BasePath, LogsPath, ModelType,NumTrainImages)
    else:
        cropSize=128
        #range for perturbing the corners to get the homographies [-rho,+rho]
        rho=16
        #resize shape
        resize=(320,240)
        #number of train sets to be made
        numTrainData=128
        #limit of the number of the images in the train set
        numImagesLimit=5000
        #file path
        filePath=BasePath+'/Train/'
        if not os.path.exists(BasePath+'/supervised/'):
            os.makedirs(BasePath+'/supervised/')
        saveDest=BasePath+'/supervised/'
        
        generateImagesSupervised(cropSize,rho,resize,numTrainData,numImagesLimit,filePath,saveDest)
        
        
        print('===============Generated'+str(numTrainData)+'=========================')

        ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 128, 128, 2))
        LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 8)) # OneHOT labels
        BasePath=saveDest
        TrainOperationSupervised(ImgPH, LabelPH, DirNamesTrain, TrainLabels,40000, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, ModelType,NumTrainImages=NumTrainImages)
        
    
if __name__ == '__main__':
    main()
 

