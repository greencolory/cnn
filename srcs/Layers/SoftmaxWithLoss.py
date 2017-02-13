#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 13:06:22 2017

@author: take
"""
from Layer import Layer
import numpy as np

class SoftmaxWithLoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        self.y = self.softmax(x)
        self._loss = self.calc_cross_entropy_err(self.y,self.t)
        
    def backward(self,dout=1):
        
        #one hot vectorの場合
        if self.y.size == self.t.size:
            dy = (self.y - self.t) / self.t.shape[0]
        else:
            dy = self.y.copy()
            dy[np.arrange(self.t.shape[0]),self.t] -= 1
            dy = self.y / self.t.shape[0]

        self.back_layer.backward(dy)

    def softmax(self,x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T 
    
        x = x - np.max(x) # オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x))

    def calc_cross_entropy_err(self,y,t):
        
        if y.ndim == 1:
            t = t.reshape(1,t.size)
            y = y.reshape(1,y.size)
        
        # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
        if t.size == y.size:
            t = t.argmax(axis=1)
    
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size),t])) / batch_size
                              
    @property
    def result(self):
        return self.y

    @result.getter
    def result(self):
        return self.y
        
    @property
    def loss(self):
        return self._loss

    @loss.getter
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self,loss):
        self._loss = loss
        
    @property
    def answer_label(self):
        return self.t
    
    @answer_label.setter
    def answer_label(self,t):
        self.t = t
    