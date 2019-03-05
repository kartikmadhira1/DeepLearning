#!/usr/bin/env python

"""
Author(s):
Kartik Madhira (kmadhira@terpmail.umd.edu)
Masters in Robotics,
University of Maryland, College Park
"""


import tensorflow as tf
import sys
import numpy as np


# Don't generate pyc codes
sys.dont_write_bytecode = True

def CIFAR10Model(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """
    total_layers = 25 #Specify how deep we want our network
    units_between_stride = total_layers / 5

    layer1 = tf.layers.conv2d(inputs=Img,filters=64,kernel_size=[3,3],activation=None,padding="SAME",name='conv_'+str(0))
    layer1=tf.layers.batch_normalization(layer1)
    for i in range(5):
	for j in range(units_between_stride):
            layer1 = resUnit(layer1,j+(i*units_between_stride))
        layer1 = tf.layers.conv2d(layer1,64,[3,3],strides=[2,2],padding="SAME",name='conv_s_'+str(i))
        layer1=tf.layers.batch_normalization(layer1)
        
    top = tf.layers.conv2d(layer1,10,[3,3],activation=None,padding="SAME",name='conv_top')
    top=tf.layers.batch_normalization(top)
    prLogits=tf.reshape(top,[-1,top.shape[1:4].num_elements()])
    prSoftMax = tf.nn.softmax((prLogits))

    return prLogits, prSoftMax

    
def resUnit(input_layer,i):
    with tf.variable_scope("res_unit"+str(i)):
        part1 = tf.layers.batch_normalization(input_layer)
        part2 = tf.nn.relu(part1)
        part3 = tf.layers.conv2d(part2,64,[3,3],activation=None,padding="SAME")
        part4 = tf.layers.batch_normalization(part3)
        part5 = tf.nn.relu(part4)
        part6 = part3 = tf.layers.conv2d(part5,64,[3,3],activation=None,padding="SAME")
        output = input_layer + part6
        output=tf.nn.relu(output)
        return output
    

