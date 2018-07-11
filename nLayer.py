
"""
This is a simple implementation of an n layer Neural Network for image classification without using DL libraries.

@author: kartikmadhira
"""

"""
Y(i)= W*X(i)+b
The final goal is to iteratively calculate the gradient descent of parameters W and b for the output Y, such that 
the W and b eventually reach their local minima and we have the optimal cost function(of error)
The process is:
    1. For the the forward propagation, go on to calculate the cost function J, be feeding vectorized Y for every layer
       and feeding it to the next layer.
"""