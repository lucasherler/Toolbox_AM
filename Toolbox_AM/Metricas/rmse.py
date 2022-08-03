#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:21:09 2021

@author: bruno
"""

from _metric import Metric 
import numpy as np

class RMSE(Metric):

    @staticmethod
    def worst_case():
        return np.Inf

    @staticmethod
    def is_regression_metric():
        return True

    @staticmethod
    def is_better(val1, val2):
        return val1 < val2

    @staticmethod
    def calculate(pred, target):
        aux = pred - target
        aux = np.reshape(np.asarray(
            np.sqrt(np.mean(np.mean(aux*aux)))),
            (1, 1))
        return aux
