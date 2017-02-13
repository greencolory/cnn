#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 14:44:58 2017

@author: take
"""

import numpy as np

class Adam():
    def __init__(self,alfa=0.001,beta1=0.9,beta2=0.999,eps=1e-8):
        self.m = None
        self.v = None
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.alfa = alfa
        self.eps = eps
        
    def updateParams(self,params,grads):
        if self.m == None:
            self.m = np.zeros_like(grads)
        if self.v == None:
            self.v = np.zeros_like(grads)
        
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        #self.m += (1 - self.beta1) * (grads - self.m) #sample code
        #self.v += (1 - self.beta2) * (grads**2 - self.v) #sample code
        #m_hat = self.m / (1 - (self.beta1 ** self.t))
        #v_hat = self.v / (1 - (self.beta2 ** self.t))
        #params -= self.alfa * m_hat / (np.sqrt(v_hat) + self.eps)
        
        self.alfa = self.alfa * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        params -= self.alfa * self.m / (np.sqrt(self.v) + self.eps)
