# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Email: harikrishnannb07@gmail.com
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.matlib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from Codes import two_layer_model_train, forward_pass, backward_pass

DATA_NAME = 'bank-note-authentication'


#import the BANK NOTE AUTHENTICATION Dataset 
bank = np.array(pd.read_csv('data_banknote_authentication.txt', sep=",", header=None))


#reading data and labels from the dataset
totaldata, totallabel = bank[:,range(0,bank.shape[1]-1)], bank[:,bank.shape[1]-1]
totallabel = totallabel.reshape(len(totallabel),1)
# X = X.astype(float)
#import the IRIS Dataset from sklearn library



#Splitting the dataset for training and testing (80-20)
traindata, testdata, trainlabel, testlabel = train_test_split(totaldata, totallabel,test_size=0.2, random_state=42)




from sklearn.preprocessing import OneHotEncoder

label_encoder = OneHotEncoder(sparse=False)
#label_encoder.fit(y[0,0].to_frame())


n_i = 4
n_h_list = [5, 10, 30, 60, 90, 100, 110]
n_l = 2
learningrate = 0.291
epochs = 500
FOLD_NO= 3
KF = KFold(n_splits= FOLD_NO, random_state=42, shuffle=True) 
KF.get_n_splits(traindata)

 

F1SCORE_FINAL = []
for n_h in n_h_list:
    for TRAIN_INDEX, VAL_INDEX in KF.split(traindata):
        FSCORE_TEMP = []
        
        
        X_TRAIN, X_VAL = traindata[TRAIN_INDEX], traindata[VAL_INDEX]
        
        Y_TRAIN, Y_VAL = trainlabel[TRAIN_INDEX], trainlabel[VAL_INDEX]
    
        Y_TRAIN_ONE_HOT = label_encoder.fit_transform(Y_TRAIN).T
        
        m = X_TRAIN.shape[0]
        f1score_total = np.zeros(epochs)
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
            Z1, A1, Z2, A2, Y_PRED = forward_pass(X_VAL.T,W1_total[:,:, rows],b1_total[:,:, rows],W2_total[:,:, rows],b2_total[:,:, rows], k1_total[:,:, rows])
            
         
            F1SCORE = f1_score(Y_VAL, Y_PRED, average="macro")
            f1score_total[rows] = F1SCORE
        FSCORE_TEMP.append(np.max(f1score_total))
    F1SCORE_FINAL.append(np.mean(FSCORE_TEMP))
    print(F1SCORE_FINAL)

import os
PATH = os.getcwd()
RESULT_PATH = PATH + '/hyperparamter-tuning/'+DATA_NAME+'/'


try:
    os.makedirs(RESULT_PATH)
except OSError:
    print ("Creation of the result directory %s not required" % RESULT_PATH)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH)

np.save(RESULT_PATH+"/n_h.npy", np.array(n_h_list[np.argmax(F1SCORE_FINAL)]) ) 
np.save(RESULT_PATH+"/n_h_list.npy", n_h_list )
np.save(RESULT_PATH+"/F1_score_ht.npy", F1SCORE_FINAL )
np.save(RESULT_PATH+"/learningrate.npy", learningrate )


#my_xticks = ['T1', 'T2', 'T3', 'T4', 'T5']
#plt.xticks(task_array, my_xticks, fontsize=25)

n_h_list_string = []
for list_num in range(0, len(n_h_list)):
    n_h_list_string.append(str(n_h_list[list_num]))
plt.figure(figsize=(15,10))    
plt.plot(F1SCORE_FINAL, '--*b', markersize = 10)

plt.xticks((range(0,len(n_h_list_string))), n_h_list_string, fontsize=45)
plt.yticks(fontsize=45)
plt.grid(True)
plt.xlabel('Number of nodes in the hidden layer', fontsize=50)
plt.ylabel('Average F1-score', fontsize=50)
plt.ylim(0.0, 1.0)
plt.tight_layout()
# plt.legend(fontsize = 35)

plt.savefig(RESULT_PATH+"/n_h_vs_f1score_"+ DATA_NAME+".eps", format='eps', dpi=300)
plt.savefig(RESULT_PATH+"/n_h_vs_f1score_"+ DATA_NAME+".jpg", format='jpg', dpi=300)
