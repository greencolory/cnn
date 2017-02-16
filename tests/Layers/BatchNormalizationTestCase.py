#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 10:04:14 2017

@author: take
"""

import unittest
import sys
import numpy as np
from unittest.mock import MagicMock

sys.path.append("../../")
sys.path.append("../../srcs/Layers/")

from srcs.Layers.BatchNormalization import BatchNormalization

class BatchNormalizationTestCase(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_case_init(self):
        opt_g = MagicMock()
        opt_b = MagicMock()
        batch_norm = BatchNormalization(opt_g,opt_b)
        
        self.assertTrue(opt_g == batch_norm.opt_g)
        self.assertTrue(opt_b == batch_norm.opt_b)
        self.assertEqual(batch_norm.eps,1e-8)
        self.assertEqual(batch_norm.gamma.shape[0],0)
        self.assertEqual(batch_norm.beta.shape[0],0)

    def test_case_forward_param_init(self):
        opt_g = MagicMock()
        opt_b = MagicMock()
        batch_norm = BatchNormalization(opt_g,opt_b)
        batch_norm.next_layer = MagicMock()
        x = np.random.randn(5,3)
        batch_norm.forward(x)
        
        self.assertTrue((batch_norm.gamma==np.ones(x.shape[1])).all())
        self.assertTrue((batch_norm.beta==np.zeros(x.shape[1])).all())
        
    def test_case_forward(self):
        #テストデータと期待値
        x = np.random.randn(5,3)
        gamma = np.ones(x.shape[1]) + 0.001
        beta = np.zeros(x.shape[1]) - 0.1
        eps = 1e-8
        u = x.sum(axis=0) / x.shape[0]
        xu = x - u
        v = ((x - u)**2).sum(axis=0) / x.shape[0]
        sqrtvar = np.sqrt(v + eps)
        ivar = 1 / sqrtvar
        #x_hat = (x - u) / np.sqrt(v + 1e-8)
        x_hat = xu * ivar
        y = gamma * x_hat
        out = y + beta
        
        #事前準備
        opt_g = MagicMock()
        opt_b = MagicMock()
        batch_norm = BatchNormalization(opt_g,opt_b)
        batch_norm.gamma = gamma
        batch_norm.beta = beta
        
        batch_norm.next_layer = MagicMock()
        
        #フォワード確認結果.
        batch_norm.forward(x)
        self.assertTrue((batch_norm.x==x).all())
        self.assertTrue((batch_norm.u==u).all())
        self.assertTrue((batch_norm.var==v).all())
        self.assertTrue((batch_norm.xu==xu).all())
        self.assertTrue((batch_norm.sqrtvar==sqrtvar).all())
        self.assertTrue((batch_norm.ivar==ivar).all())
        self.assertTrue((batch_norm.x_normalized==x_hat).all())
        self.assertTrue((batch_norm.next_layer.method_calls[0][1]==out).all())
        
        #バックワード確認結果.
        dout = np.random.randn(5,3)
        dbeta = dout.sum(axis=0)
        dgamma_x = dout
        dgamma = (dgamma_x * x_hat).sum(axis=0)
        dx_hat = dgamma_x * gamma
        dxu1 = dx_hat * ivar
        divar = (dx_hat * xu).sum(axis=0)
        dsqrtvar = -1 * divar / (sqrtvar**2)
        dvar = (0.5 / np.sqrt(v + eps)) * dsqrtvar
        dsq = np.ones_like(x)
        N = dsq.shape[0]
        dsq = dsq * dvar / N
        dxu2 = 2 * xu * dsq
        dx1 = dxu1 + dxu2
        du = -1 * (dxu1 + dxu2).sum(axis=0)
        dx2 = np.ones_like(x)
        dx2 = dx2 * du / N
        dx = dx1 + dx2
        
        batch_norm.back_layer = MagicMock()
        batch_norm.backward(dout)
        
        self.assertTrue((batch_norm.dgamma==dgamma).all())
        self.assertTrue((batch_norm.dbeta==dbeta).all())
        self.assertTrue((batch_norm.back_layer.method_calls[0][1]==dx).all())
        
    def test_case_updateParam(self):
        pass
        
if __name__ == "__main__":
    unittest.main()
