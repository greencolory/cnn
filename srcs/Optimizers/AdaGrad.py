#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 22:11:13 2017

@author: take
"""
import numpy as np

class AdaGrad():
    def __init__(self,delta=1e-7,rate=0.01):
        self.delta = delta
        self.rate = rate
        self.h = None
        
    def updateParams(self,params,grads):
        if self.h == None:
            self.h = np.zeros_like(grads)
        self.h += grads * grads
        params -= self.rate * grads / (np.sqrt(self.h) + self.delta)
            