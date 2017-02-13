#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:40:29 2017

@author: take
"""

import unittest
import sys,os
sys.path.append(os.pardir)
sys.path.append("../../")

from TwoLayerNet import TwoLayerNet
from datasetDP.mnist import load_mnist

import numpy as np

class TwoLayerNetTestCase(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_case_grad(self):
        (x_train,t_train) , (x_test,t_test) = \
        load_mnist(normalize = True,one_hot_label=True)
        
        net = TwoLayerNet(784,10,100)
        batch_mask = np.random.choice(x_train.shape[0],3)
        x = x_train[batch_mask]
        t = t_train[batch_mask]

        self.assertTrue(net.CheclGradEqualNumDiff(x,t))
        

if __name__ == "__main__":
    unittest.main()
        