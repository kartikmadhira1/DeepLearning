
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
#limit the train dataset
data_limit=10000

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

#convert row matrices to images
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

#normalize raw data
def normalize(train_data,data_limit):
    mean=np.mean(train_data,axis=1,keepdims=True)
    normalized=(mean-train_data)/255
    #this is the normalized data
    return normalized[:,:int(data_limit)]

def relu(z):
    A=np.maximum(0,z)
    return A

def sigmoid(z):
    A=1/(1+np.exp(z))
    return A
 
#initialize the weights and bias randomly to avoid zero initialization through out network
#n_x -> input layer dims
#n_h -> hidden layer size
#n_y -> output layer dims
#L -> desired number of layers
def init_weight_bias(hidden_l_len):
    #W1,b1..Wn,bn are going to be weights and biases in each layer
    W=dict()
    b=dict()
    L=len(hidden_l_len)
    for a in range(1,L):
        #Wn.shape=(n_h,n_x)
        #bn.shape=(n_h,m)
        W['W'+str(a)]=np.random.randn(hidden_l_len[a],hidden_l_len[a-1])*0.01
        b['b'+ str(a)]=np.random.randn(hidden_l_len[a],1)*0.01
        print(W['W'+str(a)].shape,b['b'+str(a)].shape)    



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