#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:23:23 2021

@author: bruno
"""

from rmse import RMSE
import numpy as np

class RRMSE(RMSE):
    @staticmethod
    def calculate(pred, target):
        aux = pred - target
        mean = np.mean(target, axis=0)
        den = mean - target
        den = np.sum(den*den, axis=0)
        num = np.sum(aux*aux, axis=0)
        aux = np.sqrt(num/den)
        return aux
