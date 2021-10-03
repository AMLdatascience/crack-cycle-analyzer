# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:01:09 2021

@author: Hong Sung Uk
"""

import pandas as pd
import numpy as np
from glob import glob

## 샘플 별로 계산
def sensor_life_maker(path):
    sensor_data_list = glob(str(path)+'//*.txt')
    
    
    all_sensor_life = pd.DataFrame([])
    for i in range(len(sensor_data_list)):
        sensor_r_data = pd.read_csv(sensor_data_list[i], encoding='euc-kr')
        sensor_r_data[['time', 'resistance', 'delx']] = pd.DataFrame(sensor_r_data[sensor_r_data.columns[0]].str.split('\t', 2).tolist())
        sensor_r_data = sensor_r_data.drop(['delx', sensor_r_data.columns[0]], 1)
        sensor_r_data['sensor_no'] = i+1
        sensor_r_data = sensor_r_data.astype('float')
        
        end_point_detect = pd.DataFrame([])
        for i in range(1, len(sensor_r_data)):
            if sensor_r_data["resistance"][i] >= sensor_r_data["resistance"][i-1]*10 :
                end_point_detect = end_point_detect.append([100])
                
            else :
                end_point_detect = end_point_detect.append([0])
                
        end_point_detect = end_point_detect.reset_index(drop=True)   
        end_point_detect[end_point_detect.idxmax()[0]:len(end_point_detect)] = end_point_detect[end_point_detect.idxmax()[0]:len(end_point_detect)].replace(0, np.NAN)
        
        sensor_life = pd.concat([sensor_r_data, end_point_detect], axis=1).dropna(axis=0)
        sensor_life.rename(columns={0:"life"}, inplace = True)
        
        sensor_life['life'] = 100/np.log(sensor_life['life'].idxmax()+1)*np.log(sensor_life['time'].values) # 센서가 끊어지는 순간의 전 스탭까지를 라이프 사이클로 정의 한 후, 그 순간까지의 cycle에 log 함수를 적용하여 수명을 추정함 (가정임)
        sensor_life['life_round'] = round(sensor_life['life'],-1) # 반올림을 이용하여 라이프 사이클을 10 단위의 값으로 변환함
        all_sensor_life = pd.concat([all_sensor_life, sensor_life])
        
    return all_sensor_life
    
path = 'data' # 경로
sensors_life = sensor_life_maker(path).reset_index(drop=True)    # 데이터 프레임 형식으로 모든 센서의 라이프 사이클이 정의됨

## 이미지에 따른 데이터 셋 추출 (2사이클 마다 계측을 실시한다고 가정함)
sensors_life_extract = pd.DataFrame([])
for i in range(len(sensors_life)):
    if sensors_life['cycle'][i]%2==0 : # 짝수 판단
        sensors_life_extract = sensors_life_extract.append(sensors_life.iloc[i])

# 여기 까지 한 후 sensors_life_extract에서 life 나 life_round를 학습시 y값으로 두고서 학습을 하면 됨