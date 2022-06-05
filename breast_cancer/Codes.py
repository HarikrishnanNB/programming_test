# -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:10:02 2022

@author: harik
"""


import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib

import matplotlib.pyplot as plt

def ada_act(x, k1_1, k1_0, play):
    a = np.multiply(k1_1, x) + k1_0
    if (play == "forward"):
        return a
    elif (play =="backward"):
        return k1_1
    


def forward_pass(X,W1,b1,W2,b2, k1):
    Z1 = np.dot(W1,X) + b1
    A1 = ada_act(Z1, k1[0,0], k1[1,0], play = "forward")
    Z2 = np.dot(W2,A1) + b2
    A2 = softmax(Z2)
    
    prediction = np.argmax(A2, axis=0)

    
    return Z1, A1, Z2, A2, prediction


def backward_pass(X, y, W1,b1,W2,b2, k1, Z1, A1, Z2, A2, dk1, m):
    
    dW2 = (1.0/m) * np.dot((A2 - y), A1.T)
    db2 = (1.0/m) * np.sum((A2 - y),axis = 1,keepdims = True)
    dk1_1 = (1.0/m) * np.sum(np.sum(np.multiply(np.dot(W2,Z1), A2), axis = 1,keepdims = True))
    dk1_0 = (1.0/m) * np.sum(np.sum(np.dot(W2.T, A2), axis = 1,keepdims = True))
    
    dk1[0, 0] = dk1_1
    dk1[1, 0] = dk1_0
    dW1 = (1.0/m) * np.dot(np.multiply(np.dot(W2.T,(A2-y)),ada_act(A1, k1[0,0], k1[1,0],play="backward")), X.T) 
    db1 = (1.0/m) * np.sum(np.multiply(np.dot(W2.T,(A2-y)),ada_act(A1, k1[0,0], k1[1,0],play="backward")),axis = 1,keepdims = True)
    
    return dW2, db2, dk1, dW1, db1

def softmax(x):
    val = np.exp(x)
    return val / val.sum()


def two_layer_model_train(X, y, W1, b1, W2, b2, k1, dk1, epochs, m, learningrate):
    
    
    loss_val = np.zeros((epochs, 1))
    # For storing the learnable parameters for all epochs
    W1_total = np.zeros((W1.shape[0], W1.shape[1], epochs))
    W2_total = np.zeros((W2.shape[0], W2.shape[1], epochs))
    b1_total = np.zeros((b1.shape[0], b2.shape[1], epochs))
    b2_total = np.zeros((b2.shape[0], b2.shape[1], epochs))
    k1_total = np.zeros((k1.shape[0], k1.shape[1], epochs))


    
    
    for numiter in range(epochs):
        # learningrate = learningrate+ numiter/(epochs+10000)
    # Forward Pass    
        Z1, A1, Z2, A2, prediction = forward_pass(X, W1, b1, W2, b2, k1)
    # Categorical Cross Entropy Loss Function
        L = (1.0/m) * -np.sum(np.multiply(y,np.log(A2)))
    # Derivative of Loss Function with Learnable Parameters
        dW2, db2, dk1, dW1, db1 = backward_pass(X, y, W1, b1, W2, b2, k1, Z1, A1, Z2, A2, dk1, m)
    # Gradient Descent Algorithm for updating learnable parameteres.
        W1 = W1 - learningrate * dW1
        b1 = b1 - learningrate * db1
        W2 = W2 - learningrate * dW2
        b2 = b2 - learningrate * db2
        k1 = k1 - learningrate * dk1
        print("<<<Loss for epoch: ",numiter+1," =>",L,">>>")
    
        W1_total[:,:, numiter] = W1
        W2_total[:,:, numiter] = W2
        b1_total[:,:, numiter] = b1
        b2_total[:,:, numiter] = b2
        k1_total[:, :, numiter] = k1
        loss_val[numiter, 0] = L
        
        
    return loss_val, W1_total, W2_total, b1_total, b2_total, k1_total