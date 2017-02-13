#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 20:27:52 2017

@author: take
"""

import numpy as np
from Layer import Layer

class Affine(Layer):
    def __init__(self,w,b,opt_w,opt_b):
        self.w = w
        self.b = b
        self.dw = None
        self.db = None
        self.opt_w = opt_w
        self.opt_b = opt_b
        super().__init__()
        
    def forward(self,x):
        #self._forward(x)
        
        self.x = x
        out = np.dot(self.x,self.w) + self.b
        self.next_layer.forward(out)
        
    def backward(self,dout):
        #self._backward(dout)
        
        self.dw = np.dot(self.x.T,dout)
        self.dx = np.dot(dout,self.w.T)
        self.db = np.sum(dout,axis=0)
        if self.back_layer:           
            self.back_layer.backward(self.dx)
        
    def updateParam(self):
        self.opt_w.updateParams(self.w,self.dw)
        self.opt_b.updateParams(self.b,self.db)

    def _forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.w) + self.b

        self.next_layer.forward(out)

    def _backward(self, dout):
        self.dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        self.dx = self.dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        if self.back_layer:           
            self.back_layer.backward(self.dx)
