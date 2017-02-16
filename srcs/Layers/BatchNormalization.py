#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 20:27:52 2017

@author: take
"""

import numpy as np
from Layer import Layer

class BatchNormalization(Layer):
    def __init__(self,opt_g,opt_b,eps = 1e-8):
        self.eps = eps
        super().__init__()
        self.gamma = np.array([])
        self.beta = np.array([])
        self.opt_g = opt_g
        self.opt_b = opt_b
    def forward(self,x):
        self.x = x
        
        if self.gamma.shape[0] == 0:
            self.gamma = np.ones(x.shape[1])
        if self.beta.shape[0] == 0:
            self.beta = np.zeros(x.shape[1])
        
        self.u = x.mean(axis=0)
        self.sq = (x - self.u)**2
        self.var = ((self.sq).sum(axis=0)) / x.shape[0]
        self.xu = x - self.u
        self.sqrtvar = np.sqrt(self.var + self.eps)
        self.ivar = 1 / self.sqrtvar
        self.x_normalized = self.xu * self.ivar
        y = self.gamma * self.x_normalized
        out = y + self.beta
        
        self.next_layer.forward(out)
        
    def backward(self,dout):
        
        self.dbeta = dout.sum(axis=0)
        dy = dout
        self.dgamma = (dy * self.x_normalized).sum(axis=0)
        dx_normalized = dy * self.gamma
        dxu1 = dx_normalized * self.ivar
        divar = (dx_normalized * self.xu).sum(axis=0)
        dsqrtvar = -1 * divar / (self.sqrtvar**2)
        dvar = (0.5 / np.sqrt(self.var + self.eps)) * dsqrtvar
        dsq = np.ones_like(self.sq)
        dsq = dsq * dvar / dsq.shape[0]
        dxu2 = 2 * self.xu * dsq
        dx1 = dxu1 + dxu2
        du = -1 * (dxu1 + dxu2).sum(axis=0)
        dx2 = np.ones_like(self.x)
        dx2 = dx2 * du / dx2.shape[0]
        out = dx2 + dx1
        self.back_layer.backward(out)
        
    def updateParam(self):
        self.opt_g.updateParams(self.gamma,self.dgamma)
        self.opt_b.updateParams(self.beta,self.dbeta)
