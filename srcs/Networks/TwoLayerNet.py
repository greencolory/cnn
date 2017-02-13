#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 20:54:00 2017

@author: take
"""

import sys
sys.path.append("../")

from common.gradient import numerical_gradient
from Affine import Affine
from ReLU import ReLU
from SoftmaxWithLoss import SoftmaxWithLoss
import numpy as np

class TwoLayerNet:
    def __init__(self,input_size,output_size,hidden_size,weight=0.01):
        self.create_two_layer_net(weight)
        
    def create_two_layer_net(self,input_size,output_size,hidden_size,weight=0.01):
        self.af_layers = []
        w1 = weight * np.random.randn(input_size,hidden_size)
        b1 = np.zeros(hidden_size)
        w2 = weight * np.random.randn(hidden_size,output_size)
        b2 = np.zeros(output_size)
        
        af1 = Affine(w = w1,b = b1)
        af2 = Affine(w = w2,b = b2)
        lu1 = ReLU()
        lu2 = ReLU()
        out = SoftmaxWithLoss()
        
        #ネットワーク構築
        af1.next_layer = lu1
        lu1.back_layer = af1
        lu1.next_layer = af2
        af2.back_layer = lu1
        af2.next_layer = lu2
        lu2.back_layer = af2
        lu2.next_layer = out
        out.back_layer = lu2
        
        self.start_layer = af1
        self.af_layers.append(af1)
        self.af_layers.append(af2)
        self.end_layer = out
        
    def predict(self,x,t):
        #正解ラベルをセットする.
        self.end_layer.answer_label = t
        
        #予測
        self.start_layer.forward(x)
        return (self.end_layer.result,self.end_layer.loss)
    
    def calc_loss(self,x,t):
        (result,loss) = self.predict(x,t)
        return loss
        
    def calc_accuracy(self,x,t):
        (result,loss) = self.predict(x,t)
        ans = np.argmax(result,axis=1)
        if t.ndim == 2:
            t = np.argmax(t,axis=1)
        
        a = np.sum(ans==t) / float(x.shape[0])

        return a
                
    def CheclGradEqualNumDiff(self,x,t,error_range=1e-6):
        num_diffs = []
        grads = []
        func = lambda w:self.calc_loss(x,t)
        
        #数値微分
        for layer in self.af_layers:
            num_diffs.append(numerical_gradient(func,layer.w))
            num_diffs.append(numerical_gradient(func,layer.b))
        
        #誤差逆伝搬
        (result,loss) = self.predict(x,t)
        self.end_layer.backward()
        for layer in self.af_layers:
            grads.append(layer.dw)
            grads.append(layer.db)
        
        #微分と勾配の誤差を比較する.
        less_than_range = []
        for num_diff,grad,i in zip(num_diffs,grads,range(len(num_diffs))):
            error = np.absolute(grad - num_diff)
            less_than_range.append((error < error_range).all())

        #debug
        print("result:")
        print(less_than_range)
        return all(less_than_range)
        
    def learn(self,x,t):
        
        (result,loss) = self.predict(x,t)
        self.end_layer.backward()
        #self.af1.updateParam()
        #self.af2.updateParam()
        for layers in self.af_layers:
            layers.updateParam()
        return loss
