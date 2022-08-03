#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:25:52 2021

@author: bruno
"""

import numpy as np
from _metric import Metric 

class MAPE(Metric):
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
    def calculate(forecasts, targets):
        if isinstance(targets, list):
            targets = np.array(targets)
        if isinstance(forecasts, list):
            forecasts = np.array(forecasts)
        return np.nanmean(np.abs(np.divide(np.subtract(targets, forecasts), targets))) * 100