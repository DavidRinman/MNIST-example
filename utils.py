#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 16:20:49 2021

@author: david
"""

import numpy as np

def onehot(arr):
  '''
    Transforms digit-representation of the correct result to a one-hot 
    representation, with all 0's except for the index of the correct class
  '''
  n = len(arr)
  ret = np.zeros((n, 10))
  ret[np.arange(n), arr] = 1
  return ret