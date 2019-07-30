# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:15:50 2019

@author: Tish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score


class knn():
    def fit(self, xtrain, ytrain, k, tree=True):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.k = k
        self.tree = tree
        self.correct = 0
        
    def dist(self, a, b):
        return np.linalg.norm(a-b)
    
    def closest(self, row, k):
        best_dist = self.dist(row, self.xtrain[0])
        best_indx = 0
        
        for i in range(self.k, len(self.xtrain)):
            dist = self.dist(row, self.xtrain[i])
            if dist < best_dist:
                best_dist = dist
                best_indx = i
            #if i % 100 == 0:
                #print("Iteration ", i)
        #print(self.ytrain[best_indx])
        return self.ytrain[best_indx]
    
    def predictKD(self, xtest, k):
        self.predictions = []
        self.kdtree = KDTree(self.xtrain, leaf_size=40)
        dist, ind = self.kdtree.query(xtest, k=self.k)
        self.predictions = self.ytrain[ind[:,0]]
        self.predictions = np.squeeze(self.predictions)
        return self.predictions        
        
    def predict(self, xtest, k):
        self.predictions = []
        for row in xtest:
            label = self.closest(row, k)
            self.predictions.append(label)
        return self.predictions
    
    def accuracy_score(self, ytrue):
        self.correct = 0
        for i in range(len(ytrue)):
            if ytrue[i] == self.predictions[i]:
                self.orrect += 1
        return (self.correct/float(len(ytrue))) * 100.0
    
    def get_results(self, ylabel):
        self.ylabel = ylabel
        size = len(self.ylabel)
        conf = confusion_matrix(self.ylabel, self.predictions)
        plt.figure(0).clf()
        plt.imshow(conf)
        print(classification_report(self.ylabel, self.predictions))
        fpr = (len(ylabel) - self.correct)/ float(len(ylabel))
        tpr = self.correct/float(len(ylabel))
        plt.figure(1).clf()
        plt.scatter(fpr, tpr, marker='o', label='KNN ROC point')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('Log_ROC')
        plt.show()
    
def main(csvX, csvY, k, test_size=0.8):
    dfA = pd.read_csv(csvX)
    dfy = pd.read_csv(csvY)
    dfA = dfA[1:]
       
    names = dfA.columns 
    if 'domain1' in names:
        dfA = dfA.drop(columns=['domain1'])
        dfA = dfA.drop(columns=['state1'])
    else:
        dfA = dfA.drop(columns=['state1'])
        dfA = dfA.drop(columns=['custAttr2'])
 
    dfA = pd.get_dummies(dfA)
    
    X = np.array(dfA)
    y = np.array(dfy)
    pca = PCA(n_components=17)
    pca_vals = pca.fit_transform(X)
    V = pca.components_
    pca_X = np.matmul(X, V[:6, :].T)   
    scaler = StandardScaler()        
    pca_X = scaler.fit_transform(pca_X)

    mask = np.random.rand(len(X)) < test_size
    
    idx = np.random.permutation(list(range(X.shape[0])))
    if pca:
        pca_X = pca_X[idx, :]
        y = y[idx]
        trX = pca_X[mask]
        testX = pca_X[~mask]
        trY = y[mask]
        testY = y[~mask] 
    else:
        X = X[idx, :]
        y = y[idx]
        trX = X[mask]
        testX = X[~mask]
        trY = y[mask]
        testY = y[~mask] 
    
    ###### 3.1 ###### 10 Random 80-20 splits for odd k from 3-23
    if test_size == 0.9:
        classifier = knn()
        classifier.fit(trX, trY, k)
        train_preds = classifier.predictKD(trX, k)
        print("For k value, ", k, ": ", accuracy_score(trY, train_preds)*100, "\tTrain Set")
        
        predictions = classifier.predictKD(testX, k)   
        print("For k value, ", k, ": ", accuracy_score(testY, predictions)*100, "\tTest Set")
        classifier.get_results(testY)
    
    ###### 3.2 and 3.3 ######
    else:
        classifier = knn()
        classifier.fit(trX, trY, k)
        train_preds = classifier.predictKD(trX, k)
        print("For k value, ", k, ": ", accuracy_score(trY, train_preds)*100, "\tTrain Set")
        classifier.get_results(trY)
        
        predictions = classifier.predictKD(testX, k)   
        print("For k value, ", k, ": ", accuracy_score(testY, predictions)*100, "\tTest Set")
        classifier.get_results(testY)
    
  
if __name__ == '__main__':
    for k in range(3,23,2):
        main(k, test_size=0.9)
    
    main(5, 0.8)