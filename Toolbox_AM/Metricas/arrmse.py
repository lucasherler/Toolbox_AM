#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:23:53 2021

@author: bruno
"""

from _rrmse import RRMSE
import numpy as np

class ARRMSE(RRMSE):
    @staticmethod
    def calculate(pred, target):
        aux = np.mean(RRMSE.calculate(pred, target))
        return aux
