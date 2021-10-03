# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:53:26 2021

@author: AML_MASTER
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from glob import glob
from PIL import Image

seed = 7777
## Train data (센서 변형 후의 이미지) 불러옴
img_list = glob('archive//Positive//*.jpg')
image_shape = (64, 64)
train_img_list, val_img_list = train_test_split(img_list, test_size=0.1, random_state=seed)

def img_to_np(fpaths, resize=True):  
    img_array = []
    for fname in fpaths:
      try:
        img = Image.open(fname).convert('RGB')
        if(resize): img = img.resize(image_shape)
        img_array.append(np.asarray(img))
      except:
        continue
    images = np.array(img_array)
    return images

#image1.transpose(Image.FL) 이미지 좌우 반전 구현해야함

x_train = img_to_np(train_img_list)
x_train = x_train.astype(np.float32) / 255.

x_val = img_to_np(val_img_list)
x_val = x_val.astype(np.float32) / 255.

## Outlier score 불러옴
os_list = np.array(range(0,20000)) #수정필요
x_train_os, x_val_os = train_test_split(os_list, test_size=0.1, random_state=seed)

## 모델 빌드
IMG_SHAPE = image_shape + (3,)

base_model = tf.keras.applications.efficientnet.EfficientNetB7(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=IMG_SHAPE, pooling=None, classes=None,
    classifier_activation=None)

#image_batch, label_batch = next(iter(x_train))
#feature_batch = base_model(image_batch)

base_model.trainable = False
#base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
#feature_batch_average = global_average_layer(feature_batch)
#print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1)
#prediction_batch = prediction_layer(feature_batch_average)

inputs = tf.keras.Input(shape=IMG_SHAPE)
inputs2 = tf.keras.Input(shape=1)
x = base_model(inputs, training=False)
x = global_average_layer(x)
x = tf.keras.layers.concatenate((x, inputs2))
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model([inputs, inputs2], outputs)

base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse', metrics=['accuracy'])

history = model.fit((x_train, x_train_os), np.array(range(0,18000)), epochs=20, validation_data=((x_val, x_val_os), np.array(range(0,2000))))