#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:22:25 2021

@author: bruno
"""

from _metric import Metric 
import numpy as np

class ACCURACY(Metric):

    @staticmethod
    def worst_case():
        return 0

    @staticmethod
    def is_regression_metric():
        return False

    @staticmethod
    def calculate(pred, target):
        predMax = np.argmax(pred, axis=1)
        targetMax = np.argmax(target, axis=1)
        value = np.sum(predMax == targetMax)/predMax.size
        return value

    @staticmethod
    def is_better(value1, value2):
        return value1 > value2
    
    
a = ACCURACY()