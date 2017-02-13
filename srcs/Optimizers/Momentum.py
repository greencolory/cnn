#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 21:37:21 2017

@author: take
"""

import numpy as np

class Momentum():
    def __init__(self,rate=0.1,resistor_rate=0.9):
        self.rate = rate
        self.resistor_rate = resistor_rate
        self.v = None
        
    def updateParams(self,params,grads):
        if self.v == None:
            self.v = np.zeros_like(grads)
        
        self.v = self.resistor_rate * self.v - self.rate * grads
        params += self.v
    