
"""
This is a simple implementation of an n layer Neural Network for image classification without using DL libraries.

@author: kartikmadhira
"""

"""
Y(i)= W*X(i)+b

The final goal is to iteratively calculate the gradient descent of parameters W and b for the output Y, such that 
the W and b eventually reach their local minima and we have the optimal value for the cost function.

The process is:
    1.  Load images dataset and set up pre-processing of images.
    1.  For the the forward propagation, go on to calculate the cost function J, by feeding vectorized Y for every layer
        to every subsequent layer. 
    2.  The function to map the values from [0,inf] to [0,1], also known as activation function will be tanh(z) for every
        hidden layer and sigmoid for the last layer. Subsequently maintain a dictionary to keep the values of W,b and Z
        for every layer stored for later use in backward propagation
    3.  For the backward step, the goal is to find dW/dJ and db/dJ for the gradient calculation. We do this by applying
        chain rule for dw[l]/dJ and db[l]/dJ calculation, where l is the layer.
    4.  Finally run a loop to descend from the gradients where we find the least values of W and b, with following eqs:
            w[i]=w[i]-dW/dJ
            b[i]=b[i]-db/dJ
"""
import numpy as np
import matplotlib.pyplot as plt

l_dict={0:'airplane',1:'automobile',2:'bird',3:'car',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
root_path='/Users/kartikmadhira/DL_datasets/cifar-10-batches-py/data_batch_'

def unpickle(file):
    import _pickle
    with open(file, 'rb') as fo:
        dict = _pickle.load(fo, encoding='bytes')
    return dict

def load_data(path):
    train_data=np.zeros((3072,1),dtype='int').T
    train_labels=[]
    for a in range(1,6):
        path=root_path+str(a)
        #unpickle the path data and get dictionary
        data=unpickle(path)
        #get data and labels 
        t_data=data[b'data']
        t_labels=data[b'labels']
        train_data=np.vstack((train_data,t_data))
        train_labels+=t_labels
    train_data=train_data[1:]
    #print(len(train_data),len(train_labels))
    return (train_data,train_labels)


def m_to_img(train_data):
    #seperating R,G,B channels
    R=train_data[:,:1024]
    B=train_data[:,1024:2048]
    G=train_data[:,2048:]
    #dstack is depth stack 
    images_data=np.dstack((R,G,B))
    images_data=np.reshape(images_data,(50000,32,32,3))
    print(images_data.shape)
    return images_data
    
def main():
    ind=10393
    train_data,train_labels=load_data(root_path)
    images_dataset=m_to_img(train_data)
    plt.figure(figsize=(3,3))
    plt.imshow(images_dataset[ind])
    print(l_dict[train_labels[ind]])
    plt.pause(10)

if(__name__=='__main__'):
    main()