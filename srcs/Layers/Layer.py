#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 20:32:13 2017

@author: take
"""

class Layer:
    def __init__(self):
        self.__next_layer = None
        self.__back_layer = None

    @property
    def next_layer(self):
        return self.__next_layer

    @next_layer.setter
    def next_layer(self,layer):
        self.__next_layer = layer
        
    @next_layer.getter
    def next_layer(self):
        return self.__next_layer
        
    @property
    def back_layer(self):
        return self.__back_layer

    @back_layer.setter
    def back_layer(self,layer):
        self.__back_layer = layer
    
    @back_layer.getter
    def back_layer(self):
        return self.__back_layer

    def forward(self,x):
        pass
    def backward(self,dout):
        pass
