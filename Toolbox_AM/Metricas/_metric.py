#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:27:00 2021

@author: bruno
"""

import abc
#import numpy as np

class Metric(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def calculate(pred, target):
        pass

    @staticmethod
    @abc.abstractmethod
    def worst_case(pred, target):
        pass

    @staticmethod
    @abc.abstractmethod
    def is_regression_metric(pred, target):
        pass

    @staticmethod
    @abc.abstractmethod
    def is_better(val1, val2):
        pass
    

if __name__ == "__main__":
    pass