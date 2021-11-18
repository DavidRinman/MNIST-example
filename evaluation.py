#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 09:05:12 2021

@author: david
"""

from tensorflow import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils import onehot

model_name = "trained_model_nodrop"

model = keras.models.load_model(model_name)
model.summary()

_, (test_x, test_y) = keras.datasets.mnist.load_data()
test_y_oh = onehot(test_y)

history = np.load(model_name+"/history.npz")
metrics = model.evaluate(x=test_x, y=test_y_oh, return_dict=True)
predictions = model.predict(x=test_x)

# plot history
epochs = np.arange(len(history['loss']))+1
fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.plot(epochs, history['loss'], label='Training data', marker='o')
ax.plot(epochs, history['val_loss'], label='Validation data', marker='o')
ax.set(xlabel='Epoch', ylabel='Categorical Crossentropy', title='Training history')
ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax.legend()
ax.grid()

# index false classifications
pred = np.argmax(predictions, axis=1)
idx_wrong = np.where(pred != test_y)[0]
idx_ex = np.random.choice(idx_wrong, size=9, replace=False)

# plot some failed classifications
fig, axs = plt.subplots(3,3, figsize=(8,8))
fig.suptitle('Examples of incorrect classifications')
for i, ax in enumerate(axs.ravel()):
  img = test_x[idx_ex[i]]
  ax.imshow(img, cmap='gray')
  ax.axes.xaxis.set_ticklabels([])
  ax.axes.yaxis.set_ticklabels([])