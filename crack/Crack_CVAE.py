# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 13:40:45 2021

@author: Hong Sung Uk
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Reshape, InputLayer
from alibi_detect.od import OutlierVAE
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from glob import glob
import pandas as pd
from alibi_detect.utils.saving import save_detector, load_detector
import os

path = 'model_save'

def img_to_np(fpaths, resize=True):  
    img_array = []
    for fname in fpaths:
      try:
        img = Image.open(fname).convert('RGB') #RGB
        if(resize): img = img.resize((64, 64))               
        img_array.append(np.asarray(img))
      except:
        continue
    images = np.array(img_array)
    return images
    
if os.listdir(path)[-1] == 'OutlierVAE.pickle' :
    od = load_detector("model_save")

else : 
    ## Train data (센서 변형 전의 이미지) 불러옴
    img_list = glob('archive//Negative//*.jpg')
    
    #train_img_list, val_img_list = train_test_split(img_list, test_size=0.1, random_state=2021)

    x_train = img_to_np(img_list)
    x_train = x_train.astype(np.float32) / 255.    
    x_train_lr = tf.image.flip_left_right(x_train).numpy()
    x_train_ud = tf.image.flip_up_down(x_train).numpy()
    x_train = np.concatenate((x_train, x_train_lr, x_train_ud))
    
    #x_val = img_to_np(val_img_list)
    #x_val = x_val.astype(np.float32) / 255.
    plt.imshow(x_train[0])
    plt.imshow(x_train_lr[0])
    plt.imshow(x_train_ud[0])
    
    ## CVAE 모델 빌드
    latent_dim = 1024
    
    encoder_net = tf.keras.Sequential([
        InputLayer(input_shape=(64, 64, 3)),
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
        Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')
    ])
    
    od = OutlierVAE(threshold=.005, score_type='mse', encoder_net=encoder_net, decoder_net=decoder_net,latent_dim=latent_dim)
    
    ## 학습
    od.fit(x_train,epochs=30,verbose=True)
    
    ## 저장
    save_detector(od, "model_save")

## Test 데이터 (센서 변형 후 데이터) 로드
test_img_list = glob('archive//Positive_crack_sensor8_20//*.jpg')
x_test = img_to_np(test_img_list)
x_test = x_test.astype(np.float32) / 255.

## Test 데이터 아웃라이어 확인
od_preds = od.predict(x_test, outlier_type='instance', return_feature_score=True, return_instance_score=True)

## 결과 저장 : 1. 이미지 저장
x_recon = od.vae(x_test).numpy()
scores = od_preds['data']['feature_score']
instance_ids = list(range(len(od_preds['data']['is_outlier'])))
n_instances = min(len(test_img_list), len(instance_ids))
instance_ids = instance_ids[:n_instances]
n_cols = 5
        
mpl.rcParams['font.family'] = 'sans-serif'
fig, axes = plt.subplots(nrows=n_instances, ncols=n_cols, figsize=(20, 20))
fig.suptitle(test_img_list[0].split('_',6)[2] + "'s Crack images", size = 30, y = 0.92)

n_subplot = 1
for i in range(n_instances):
    idx = instance_ids[i]
    X_outlier = x_test[idx]
    plt.subplot(n_instances, n_cols, n_subplot)
    plt.axis('off')
    if i == 0:
        plt.title('Original')
    plt.imshow(X_outlier)
    n_subplot += 1
    
    plt.subplot(n_instances, n_cols, n_subplot)
    plt.axis('off')
    if i == 0:
        plt.title('Reconstruction')
    plt.imshow(x_recon[idx])
    n_subplot += 1

    plt.subplot(n_instances, n_cols, n_subplot)
    plt.axis('off')
    if i == 0:
        plt.title('Outlier Score Channel 0')
    plt.imshow(scores[idx][:, :, 0])
    n_subplot += 1
   
    plt.subplot(n_instances, n_cols, n_subplot)
    plt.axis('off')
    if i == 0:
        plt.title('Outlier Score Channel 1')
    plt.imshow(scores[idx][:, :, 1])
    n_subplot += 1

    plt.subplot(n_instances, n_cols, n_subplot)
    plt.axis('off')
    if i == 0:
        plt.title('Outlier Score Channel 2')
    plt.imshow(scores[idx][:, :, 2])
    n_subplot += 1

plt.savefig(test_img_list[0].split('_',6)[2] + '_images', edgecolor='white', dpi=500, facecolor='white', transparent=True, bbox_inches='tight', pad_inches=0.5)
plt.show()
plt.clf()

## 결과 저장 : 2. 그래프 저장
img_cycle = pd.DataFrame()
for i in range(len(test_img_list)):
    img_cycle = img_cycle.append([int(test_img_list[i].split('_',6)[5][0:3])])
img_cycle.rename(columns={0:"cycle"}, inplace = True)
img_cycle = img_cycle.reset_index(drop=True)

plt.plot(img_cycle['cycle'].values, (od_preds['data']['instance_score']/od_preds['data']['instance_score'][0]-1)*100, 'r--o')
plt.title(test_img_list[0].split('_',6)[2] + "'s Crack Score")
plt.xlabel("Cycles", size = 14)
plt.ylabel("Crack Score", size = 14)
plt.savefig(test_img_list[0].split('_',6)[2] +'_graph', edgecolor='white', dpi=500, facecolor='white', transparent=True, bbox_inches='tight', pad_inches=0.5)
plt.show()

## 결과 저장 : 3. 스코어 데이터 저장
out_data = pd.concat([img_cycle, pd.DataFrame(od_preds['data']['instance_score'], columns=['Crack Score']).reset_index(drop=True), pd.DataFrame(od_preds['data']['instance_score']/od_preds['data']['instance_score'][0]-1, columns=['Relative Crack Score'])*100], axis=1)
out_data.to_csv(test_img_list[0].split('_',6)[2]+"'s_crack_score_score.csv", index=False)