#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.

import sys
sys.path.insert(0, "../../ncapi2_shim")
import mvnc_simple_api as mvnc
import numpy
import cv2
import matplotlib.pyplot as plt

path_to_networks = './'
path_to_images = '../../data/images/'
graph_filename = 'graph'
image_filename = path_to_images + 'nps_electric_guitar.png'

#mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 2)
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No devices found')
    quit()

device = mvnc.Device(devices[0])
device.OpenDevice()

#Load graph
with open(path_to_networks + graph_filename, mode='rb') as f:
    graphfile = f.read()

#Load preprocessing data
mean = 128 
std = 1.0/128.0 

#Load categories
categories = []
with open(path_to_networks + 'categories.txt', 'r') as f:
    for line in f:
        cat = line.split('\n')[0]
        if cat != 'classes':
            categories.append(cat)
    f.close()
    print('Number of categories:', len(categories))

#Load image size
with open(path_to_networks + 'inputsize.txt', 'r') as f:
    reqsize = int(f.readline().split('\n')[0])

graph = device.AllocateGraph(graphfile)

import time 
timeStamps=[]
imageInd=[]
for j in range(30):
	start=time.time()
	img = cv2.imread(image_filename).astype(numpy.float32)
	dx,dy,dz= img.shape
	delta=float(abs(dy-dx))
	if dx > dy: #crop the x dimension
	    img=img[int(0.5*delta):dx-int(0.5*delta),0:dy]
	else:
	    img=img[0:dx,int(0.5*delta):dy-int(0.5*delta)]
	    
	img = cv2.resize(img, (reqsize, reqsize))

	img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

	for i in range(3):
	    img[:,:,i] = (img[:,:,i] - mean) * std

	#print('Start download to NCS...')
	graph.LoadTensor(img.astype(numpy.float16), 'user object')
	output, userobj = graph.GetResult()

	top_inds = output.argsort()[::-1][:5]
	#for i in range(5):
	    #print(top_inds[i], categories[top_inds[i]], output[top_inds[i]])
	exec_time=(time.time() - start)
	timeStamps.append(exec_time)
	imageInd.append(j);
	print("--- %s seconds ---" % (time.time() - start))
mean=sum(timeStamps)/len(timeStamps)
print('Mean execution time is' +str(mean))
plt.bar(imageInd,timeStamps)
plt.text(15.5,0.15,'Mean execution time: '+str(round(mean,5))+'s',horizontalalignment='center',
     verticalalignment='center')
plt.ylabel('Execution Time (s)');
plt.xlabel('Inference Image');
plt.pause(5)
print(''.join(['*' for i in range(79)]))
graph.DeallocateGraph()
device.CloseDevice()
print('Finished')