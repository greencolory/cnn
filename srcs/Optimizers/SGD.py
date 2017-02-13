#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 21:19:29 2017

@author: take
"""

class SGD():
    def __init__(self,rate):
        self.rate = rate
    def updateParams(self,params,grads):
        params -= self.rate * grads
        