
"""
This is a simple implementation of an n layer Neural Network for image classification without using DL libraries.

@author: kartikmadhira
"""

"""
Y(i)= W*X(i)+b

The final goal is to iteratively calculate the gradient descent of parameters W and b for the output Y, such that 
the W and b eventually reach their local minima and we have the optimal value for the cost function.

The process is:
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