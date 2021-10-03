# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 13:40:45 2021

@author: Hong Sung Uk
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Layer, Reshape, InputLayer
from alibi_detect.models.losses import elbo
from alibi_detect.od import OutlierVAE
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from glob import glob

## Train data (센서 변형 전의 이미지) 불러옴
img_list = glob('archive//Negative//*.jpg')

train_img_list, val_img_list = train_test_split(img_list, test_size=0.1, random_state=2021)

def img_to_np(fpaths, resize=True):  
    img_array = []
    for fname in fpaths:
      try:
        img = Image.open(fname).convert('L') #RGB
        if(resize): img = img.resize((64, 64))
        img_array.append(np.asarray(img))
      except:
        continue
    images = np.array(img_array)
    return images

x_train = img_to_np(train_img_list)
x_train = x_train.astype(np.float32) / 255.
x_train = x_train.reshape(-1, 64, 64, 1)

x_val = img_to_np(val_img_list)
x_val = x_val.astype(np.float32) / 255.
x_val = x_val.reshape(-1, 64, 64, 1)

print(x_train.shape)
print(x_val.shape)
plt.imshow(x_train[0])

## CVAE 모델 빌드
latent_dim = 1024

encoder_net = tf.keras.Sequential([
    InputLayer(input_shape=(64, 64, 1)),
    Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
    Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
    Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu)
])

decoder_net = tf.keras.Sequential([
    InputLayer(input_shape=(latent_dim,)),
    Dense(4 * 4 * 128),
    Reshape(target_shape=(4, 4, 128)),
    Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu),
    Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu),
    Conv2DTranspose(32, 4, strides=2, padding='same', activation=tf.nn.relu),
    Conv2DTranspose(1, 4, strides=2, padding='same', activation='sigmoid')
])

od = OutlierVAE(threshold=.005, score_type='mse', encoder_net=encoder_net, decoder_net=decoder_net,latent_dim=latent_dim)

## 학습
od.fit(x_train,epochs=30,verbose=True)

## 재현
idx = 12
x = x_train[idx].reshape(1, 64, 64, 3)
x_recon = od.vae(x).numpy()

fig, axes = plt.subplots(1, 2, figsize=(16, 10))

axes[0].imshow(x.squeeze())
axes[1].imshow(x_recon.squeeze())

## Validation
od_preds = od.predict(x_val, outlier_type='instance', return_feature_score=True, return_instance_score=True)

target = np.zeros(x_val.shape[0],).astype(int)
labels = ['normal', 'outlier']
plot_instance_score(od_preds, target, labels, od.threshold)

x_recon = od.vae(x_val).numpy()
plot_feature_outlier_image(od_preds, x_val, X_recon=x_recon, max_instances=5, outliers_only=False)

## Test 데이터 (센서 변형 후 데이터) 로드
test_img_list = glob('archive//Positive_crack_sensor4_20//*.jpg')

x_test = img_to_np(test_img_list)
x_test = x_test.astype(np.float32) / 255.
x_test = x_test.reshape(-1, 64, 64, 1)
print(x_test.shape)

## Test 데이터 아웃라이어 확인
od_preds = od.predict(x_test, outlier_type='instance', return_feature_score=True, return_instance_score=True)

target = np.zeros(x_test.shape[0],).astype(int)
labels = ['normal', 'outlier']
plot_instance_score(od_preds, target, labels, od.threshold)

## Test 데이터 재현
x_recon = od.vae(x_test).numpy()
plot_feature_outlier_image(od_preds, x_test, X_recon=x_recon, n_channels=1, max_instances=10, outliers_only=False)


od_preds['data']['instance_score'].to_csv('test_data_score.csv', index=False)