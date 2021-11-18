#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 15:41:10 2021

@author: david
"""
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten, \
  Dropout

def CNN():
  inputs = Input(shape=(28,28,1))
  x = Conv2D(32, kernel_size=(3,3), strides=(1,1), activation='relu')(inputs)
  x = MaxPool2D(pool_size=(2,2))(x)
  x = Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu')(x)
  x = MaxPool2D(pool_size=(2,2))(x)
  x = Conv2D(8, kernel_size=(1,1), strides=(1,1), activation='relu')(x)
  x = Flatten()(x)
  x = Dense(64, activation='relu')(x)
  x = Dropout(0.5)(x)
  x = Dense(64, activation='relu')(x)
  x = Dropout(0.5)(x)
  x = Dense(10, activation='softmax')(x)
  outputs=x
  return keras.Model(inputs, outputs)
  