#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 15:31:51 2021

@author: david
"""

from tensorflow import keras
import numpy as np

from models import CNN
from utils import onehot

## load data
(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
train_y, test_y = onehot(train_y), onehot(test_y)

## load and compile model
model_name = "trained_model_nodrop"
model = CNN(dropout_rate=0.0)
model.summary()

opti = keras.optimizers.Adam()
loss = keras.losses.CategoricalCrossentropy()
metr = keras.metrics.CategoricalAccuracy()
model.compile(loss=loss, optimizer=opti, metrics=[metr])

## train model
es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                   restore_best_weights=True)
history = model.fit(x=train_x, y=train_y, validation_split=0.1,
                    epochs=60, batch_size=32, callbacks=[es])

## save model and history
model.save(model_name)
kwargs = {key: history.history[key] for key in history.history.keys()}
np.savez(model_name+"/history.npz", **kwargs)