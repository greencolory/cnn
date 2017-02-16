#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 20:54:00 2017

@author: take
"""

import sys
sys.path.append("../")

from common.gradient import numerical_gradient
from Layers.Affine import Affine
from Layers.ReLU import ReLU
from Layers.BatchNormalization import BatchNormalization
from Layers.SoftmaxWithLoss import SoftmaxWithLoss
import numpy as np
from Optimizers.SGD import SGD
from Optimizers.Momentum import Momentum
from Optimizers.AdaGrad import AdaGrad

class NLayerNet:
    def __init__(self,layers_size,weight=0.01):
        self.create_N_layer_net(layers_size,weight)
        
    def create_N_layer_net(self,layers_size,weight):
        #w = []
        #b = []
        self.update_layers = []
        #レイヤ生成
        layers = []
        for i in range(len(layers_size) - 1):
            w = weight * np.random.randn(layers_size[i],layers_size[i+1])
            b = np.zeros(layers_size[i+1])
            a = Affine(w = w,b = b,opt_w = SGD(0.1),opt_b = SGD(0.1))
            #a = Affine(w = w[i],b = b[i],opt_w = Momentum(rate=0.001),opt_b = Momentum(rate=0.001))
            #a = Affine(w = w[i],b = b[i],opt_w = AdaGrad(rate = 0.001),opt_b = AdaGrad(rate = 0.001))
            #a = Affine(w = w[i],b = b[i],opt_w = Adam(),opt_b = Adam())
            self.update_layers.append(a)
            layers.append(a)
            #batch normalizationを挿入する.
            batch_norm = BatchNormalization(SGD(0.1),SGD(0.1))
            self.update_layers.append(batch_norm)
            layers.append(batch_norm)
            layers.append(ReLU())
        
        layers.append(SoftmaxWithLoss())
        
        #ネットワーク構築
        #フォワード方向        
        for i in range(len(layers)-1):
            layers[i].next_layer = layers[i+1]
        #バック方向
        for i in range(len(layers)-1,0,-1):
            layers[i].back_layer = layers[i-1]
        
        self.start_layer = layers[0]
        self.end_layer = layers[-1]
        
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
        for layer in self.update_layers:
            layer.updateParam()
        return loss
