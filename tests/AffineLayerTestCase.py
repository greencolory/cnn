#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 10:04:14 2017

@author: take
"""

import unittest
import sys,os
import numpy as np
from unittest.mock import MagicMock

sys.path.append(os.pardir)

from Affine import Affine

class AffineLayerTestCase(unittest.TestCase):
    def setUp(self):
        self.w = np.array([[1.0,3.0],[4.0,5.0],[6.0,7.0]])
        self.b = np.array([2.0,4.0])
        self.x = np.array([[3.0,4.0,5.0],[5.0,6.0,7.0]])
        
    def tearDown(self):
        pass
    def test_case_forward(self):
        af = Affine(self.w,self.b)
        af.next_layer = MagicMock()
        af.forward(self.x)
        ans = np.dot(self.x,self.w)+self.b
        out = af.next_layer.method_calls[0][1]
        self.assertTrue((ans==out).all())

    def test_case_backward(self):
        af = Affine(self.w,self.b)
        af.back_layer = MagicMock()
        dy = np.array([[2,4],[5,4]])
        af.next_layer = MagicMock()
        af.back_layer = MagicMock()
        af.forward(self.x)
        
        dw = np.dot(self.x.T,dy)
        dx = np.dot(dy,self.w.T)
        db = np.sum(dy,axis=0)
        af.backward(dy)
        self.assertTrue((af.dw==dw).all())
        self.assertTrue((af.db==db).all())
        self.assertTrue((af.dx==dx).all())
        self.assertTrue((af.back_layer.method_calls[0][1]==dx).all())
        
    def test_case_updateParam(self):
        af = Affine(self.w,self.b)
        dy = np.array([[2,4],[5,4]])
        af.next_layer = MagicMock()
        af.back_layer = MagicMock()
        af.forward(self.x)
        af.backward(dy)
        w = af.w - 0.01*af.dw
        b = af.b - 0.01*af.db
        
        af.updateParam()
        self.assertTrue((af.w==w).all())
        self.assertTrue((af.b==b).all())
        
if __name__ == "__main__":
    unittest.main()
