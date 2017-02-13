#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:54:51 2017

@author: take
"""
from Layer import Layer

class ReLU(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        self.next_layer.forward(out)
        
    def backward(self,dout):
        dx = dout.copy()
        dx[self.mask] = 0
        self.back_layer.backward(dx)
        