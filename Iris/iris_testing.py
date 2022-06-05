# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 17:20:14 2022

@author: harik
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from Codes import two_layer_model_train, forward_pass, backward_pass


DATA_NAME = "Iris"

#import the IRIS Dataset from sklearn library
iris = datasets.load_iris()

#reading data and labels from the dataset
totaldata = np.array(iris.data)
totallabel = np.array(iris.target)
totallabel = totallabel.reshape(len(totallabel),1)


#Splitting the dataset for training and testing (80-20)
traindata, testdata, trainlabel, testlabel = train_test_split(totaldata, totallabel,test_size=0.2, random_state=42)




from sklearn.preprocessing import OneHotEncoder

label_encoder = OneHotEncoder(sparse=False)
#label_encoder.fit(y[0,0].to_frame())

PATH = os.getcwd()
RESULT_PATH = PATH + '/hyperparamter-tuning/'+DATA_NAME+'/'
    
n_h = np.load(RESULT_PATH+"/n_h.npy") # Number of nodes in the hidden layer 

n_i = 4 # Number of nodes in the input layer
n_l = 3 # Number of nodes in the output layer
learningrate = np.load(RESULT_PATH+"/learningrate.npy")
epochs = 500

 


Y_TRAIN = trainlabel
X_TRAIN = traindata 
Y_TRAIN_ONE_HOT = label_encoder.fit_transform(Y_TRAIN).T

m = X_TRAIN.shape[0]
f1score_total = np.zeros(epochs)
accuracy_total = np.zeros(epochs)

# Intialization of Random Initial weights and bias (A seed value is given for reproducibility)
np.random.seed(62)
W1 = np.random.randn(n_h,n_i)*0.01
np.random.seed(12)
b1 = np.random.randn(n_h,1)*0.01
np.random.seed(32)
k1 = np.random.randn(2,1)*0.01
np.random.seed(62)
W2 = np.random.randn(n_l,n_h)*0.01
np.random.seed(2)
b2 = np.random.randn(n_l,1)



# Initialization of derivative of loss function with respect to k1
dk1 = np.zeros((2,1))


loss_val, W1_total, W2_total, b1_total, b2_total, k1_total = two_layer_model_train(X_TRAIN.T, Y_TRAIN_ONE_HOT, W1, b1, W2, b2, k1, dk1, epochs, m, learningrate)



for rows in range(0, epochs):
    Z1, A1, Z2, A2, Y_PRED = forward_pass(X_TRAIN.T,W1_total[:,:, rows],b1_total[:,:, rows],W2_total[:,:, rows],b2_total[:,:, rows], k1_total[:,:, rows])
    
 
    F1SCORE = f1_score(trainlabel, Y_PRED, average="macro")
    f1score_total[rows] = F1SCORE
    accuracy_total[rows] = accuracy_score(trainlabel, Y_PRED)*100
print("Best parameter index  = ", np.argmax(f1score_total), "f1score val = ", np.max(f1score_total))

### Testing
rows = np.argmax(f1score_total)

Z1, A1, Z2, A2, Y_PRED = forward_pass(testdata.T,W1_total[:,:, rows],b1_total[:,:, rows],W2_total[:,:, rows],b2_total[:,:, rows], k1_total[:,:, rows])
    
 
F1SCORE = f1_score(testlabel, Y_PRED, average="macro")

print('TRAINING F1 SCORE =', f1score_total[rows])
print('TESTING F1 SCORE =', F1SCORE)
print('TRAIN ACCURACY = ', accuracy_total[rows])
print('TEST ACCURACY = ', accuracy_score(testlabel, Y_PRED)*100)

plt.figure(figsize=(15,10))    
plt.plot(np.arange(1, epochs+1), loss_val, '--*b', markersize = 10, label= 'Training Loss')
plt.plot(np.arange(1, epochs+1), f1score_total, '--sr', markersize = 10, label = 'Train F1-score')
#plt.xticks((range(0,len(n_h_list_string))), n_h_list_string, fontsize=45)
plt.xticks(fontsize=45)
plt.yticks(fontsize=45)
plt.grid(True)
plt.xlabel('Number of epochs', fontsize=50)
plt.tight_layout()
plt.legend(fontsize = 35)

plt.savefig(RESULT_PATH+"/training_loss_vs_epochs_"+ DATA_NAME+".eps", format='eps', dpi=300)
plt.savefig(RESULT_PATH+"/training_loss_vs_epochs_"+ DATA_NAME+".jpg", format='jpg', dpi=300)



