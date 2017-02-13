#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 14:38:50 2017

@author: take
"""
import sys
#sys.path.append(os.pardir)
sys.path.append("./Layers")
sys.path.append("./common")
sys.path.append("./Optimizers")
sys.path.append("./data")
sys.path.append("./Networks")
import time

from Networks.NLayerNet import NLayerNet

from data.mnist import load_mnist

import numpy as np

class Classifier():
    def __init__(self):
        layers_size = [784,100,10]
        self.net = NLayerNet(layers_size)
        (self.x_train,self.t_train) , (self.x_test,self.t_test) = \
        load_mnist(normalize = True,one_hot_label=True)
        
        self.BUTCH_SIZE = 100
        self.TRAIN_DATA_NUM = self.x_train.shape[0]
        self.ITERS_NUM = 10000
        self.accuracy = []
        self.accuracy_test = []


    def classify(self):
        start = time.time()
        for id in range(self.ITERS_NUM):
            batch_mask = np.random.choice(self.TRAIN_DATA_NUM,self.BUTCH_SIZE)
            x = self.x_train[batch_mask]
            t = self.t_train[batch_mask]

            self.net.learn(x,t)
            
            if id % int(self.TRAIN_DATA_NUM / self.BUTCH_SIZE) == 0:
                a = self.net.calc_accuracy(x,t)
                a_test = self.net.calc_accuracy(self.x_test,self.t_test)
                self.accuracy.append(a)
                self.accuracy_test.append(a_test)
    
        print("result accuracy train=",self.accuracy[-1])
        print("result accuracy test=",self.accuracy_test[-1])
        print("Processing Time:",time.time()-start,"[s]")
    def show_loss_graph(self):
        import matplotlib.pyplot as plt
        x = range(len(self.accuracy))
        plt.plot(x,self.accuracy,linestyle="--",label="Accuracy Train Data")
        plt.plot(x,self.accuracy_test,label="Accuracy Test Data")
        plt.legend(bbox_to_anchor=(1.05,0))
        plt.show()

if __name__ == "__main__":
    cls = Classifier()
    cls.classify()
    cls.show_loss_graph()
    