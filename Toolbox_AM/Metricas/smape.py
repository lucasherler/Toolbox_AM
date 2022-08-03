#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:26:24 2021

@author: bruno
"""

from mape import MAPE
import numpy as np

class SMAPE(MAPE):
    @staticmethod
    def calculate(forecasts, targets, type=2):
        if isinstance(targets, list):
            targets = np.array(targets)
        if isinstance(forecasts, list):
            forecasts = np.array(forecasts)
        if type == 1:
            return np.nanmean(np.abs(forecasts - targets) / ((forecasts + targets) / 2))
        elif type == 2:
            return np.nanmean(np.abs(forecasts - targets) / (np.abs(forecasts) + abs(targets))) * 100
        else:
            return np.nansum(np.abs(forecasts - targets)) / np.nansum(forecasts + targets)