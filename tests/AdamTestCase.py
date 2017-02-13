#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 10:04:14 2017

@author: take
"""

import unittest
import sys,os
import numpy as np

sys.path.append(os.pardir)

from Adam import Adam

class AdamTestCase(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_case_updateParams(self):
        w = np.array([[1.0,3.0],[4.0,5.0],[6.0,7.0]])
        dw = np.array([[0.05,0.3],[0.01,0.5],[0.2,0.3]])


        opt = Adam()
        
        #初期化期待値.
        self.assertEqual(opt.t,0)
        self.assertEqual(opt.alfa,0.001)
        self.assertEqual(opt.beta1,0.9)
        self.assertEqual(opt.beta2,0.999)
        self.assertEqual(opt.eps,1e-8)
        
        #1回目期待値(実行する前に作る).
        m_exp = np.zeros_like(w)
        v_exp = np.zeros_like(w)
        
        m_exp = 0.9 * m_exp + (1 - 0.9) * dw
        v_exp = 0.999 * v_exp + (1 - 0.999) * dw ** 2
        alfa_exp = 0.001
        alfa_exp = alfa_exp * np.sqrt(1 - 0.999 ** 1) / (1 - 0.9 ** 1)
        w_exp = w - alfa_exp * m_exp / (np.sqrt(v_exp) + 1e-8)
        
        #実行
        opt.updateParams(w,dw)
        
        self.assertEqual(opt.t,1)
        self.assertTrue((opt.m == m_exp).all())
        self.assertTrue((opt.v == v_exp).all())
        self.assertEqual(opt.alfa,alfa_exp)
        self.assertTrue((w == w_exp).all())

        #2回目期待値(実行する前に作る).        
        m_exp = 0.9 * m_exp + (1 - 0.9) * dw
        v_exp = 0.999 * v_exp + (1 - 0.999) * dw ** 2
        alfa_exp = alfa_exp * np.sqrt(1 - 0.999 ** 2) / (1 - 0.9 ** 2)
        w_exp = w - alfa_exp * m_exp / (np.sqrt(v_exp) + 1e-8)
        
        #実行
        opt.updateParams(w,dw)
        
        self.assertEqual(opt.t,2)
        self.assertTrue((opt.m == m_exp).all())
        self.assertTrue((opt.v == v_exp).all())
        self.assertEqual(opt.alfa,alfa_exp)
        self.assertTrue((w == w_exp).all())

        
if __name__ == "__main__":
    unittest.main()
