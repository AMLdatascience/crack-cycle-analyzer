# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:56:43 2021

@author: AML_MASTER
"""
import pandas as pd
from matplotlib import pyplot
import numpy as np
import decimal

# 데이터 불러오기
db = pd.read_csv('data//211003_sensor_8.txt')
db[['time', 'resistance', 'delx']] = pd.DataFrame(db[db.columns[0]].str.split('\t', 2).tolist())
db = db.drop(['delx', db.columns[0]], 1)
db = db.astype('float')

# 데이터 결측 값 보간
db_time = pd.DataFrame()
db_time['time'] = np.arange(0.01, float(db['time'].iloc[-1]), 0.01)
db = pd.merge(db, db_time, how='right')
db = db.interpolate(method="polynomial", order=2)
db = db.dropna(axis=0)
db = db.reset_index(drop=True)
db['time'] = round(db['time'],2)

# 1초 단위로 데이터 추출
db2 = pd.DataFrame([])
for i in range(len(db)):
    if db['time'][i]%1==0 : 
        db2 = db2.append(db.iloc[i])

db2 = db2.reset_index(drop=True)
db2 = round(db2)

# 이동평균 계산
db2['5mean_resistance'] = db2['resistance'].rolling(window=5, center=True, min_periods=1).mean()

# 1차 기울기 계산
temp_gradient_df = pd.DataFrame()
for i in range(1, len(db2)):
    temp_gradient = (db2['5mean_resistance'][i] - db2['5mean_resistance'][i-1])/0.01
    temp_gradient_df = temp_gradient_df.append([temp_gradient])

temp_gradient_df_r = round(temp_gradient_df,-1).reset_index(drop=True)
temp_gradient_df_r.rename(columns={0:"gradient_1"}, inplace = True)
temp_gradient_df_r_index = pd.DataFrame(np.arange(1, db2.index.max()+1, 1)).reset_index(drop=True)
temp_gradient_df_r_index.rename(columns={0:"r_index"}, inplace = True)

temp_gradient_df_r = pd.concat([temp_gradient_df_r, temp_gradient_df_r_index], axis=1)
temp_gradient_df_r = temp_gradient_df_r.set_index('r_index', drop=True)

db2 = db2.join(temp_gradient_df_r)

# 1차 기울기 계산 (2 step 전)
temp_gradient_df_2sb = pd.DataFrame()
for i in range(2, len(db2)):
    temp_gradient = (db2['5mean_resistance'][i] - db2['5mean_resistance'][i-2])/0.01
    temp_gradient_df_2sb = temp_gradient_df_2sb.append([temp_gradient])

temp_gradient_df_2sb_r = round(temp_gradient_df_2sb,-1).reset_index(drop=True)
temp_gradient_df_2sb_r.rename(columns={0:"gradient_1_2sb"}, inplace = True)
temp_gradient_df_2sb_r_index = pd.DataFrame(np.arange(2, db2.index.max()+2, 1)).reset_index(drop=True)
temp_gradient_df_2sb_r_index.rename(columns={0:"r_index"}, inplace = True)

temp_gradient_df_2sb_r = pd.concat([temp_gradient_df_2sb_r, temp_gradient_df_2sb_r_index], axis=1)
temp_gradient_df_2sb_r = temp_gradient_df_2sb_r.set_index('r_index', drop=True)

db2 = db2.join(temp_gradient_df_2sb_r)

# 1차 기울기 계산 (2 step 후)
temp_gradient_df_2sf = pd.DataFrame()
for i in range(len(db2)-2):
    temp_gradient = (db2['5mean_resistance'][i+2] - db2['5mean_resistance'][i])/0.01
    temp_gradient_df_2sf = temp_gradient_df_2sf.append([temp_gradient])

temp_gradient_df_2sf_r = round(temp_gradient_df_2sf,-1).reset_index(drop=True)
temp_gradient_df_2sf_r.rename(columns={0:"gradient_1_2sf"}, inplace = True)
temp_gradient_df_2sf_r_index = pd.DataFrame(np.arange(0, db2.index.max(), 1)).reset_index(drop=True)
temp_gradient_df_2sf_r_index.rename(columns={0:"r_index"}, inplace = True)

temp_gradient_df_2sf_r = pd.concat([temp_gradient_df_2sf_r, temp_gradient_df_2sf_r_index], axis=1)
temp_gradient_df_2sf_r = temp_gradient_df_2sf_r.set_index('r_index', drop=True)

db2 = db2.join(temp_gradient_df_2sf_r)

# 2차 기울기 계산
temp_gradient2_df = pd.DataFrame()
for i in range(1, len(db2)-1):
    temp_gradient2 = (db2['gradient_1'][i+1] - db2['gradient_1'][i])/0.01
    temp_gradient2_df = temp_gradient2_df.append([temp_gradient2])
    
temp_gradient2_df_r = round(temp_gradient2_df,-1).reset_index(drop=True)
temp_gradient2_df_r.rename(columns={0:"gradient_2"}, inplace = True)
temp_gradient2_df_r_index = pd.DataFrame(np.arange(1, db2.index.max(), 1)).reset_index(drop=True)
temp_gradient2_df_r_index.rename(columns={0:"r_index"}, inplace = True)

temp_gradient2_df_r = pd.concat([temp_gradient2_df_r, temp_gradient2_df_r_index], axis=1)
temp_gradient2_df_r = temp_gradient2_df_r.set_index('r_index', drop=True)

db2 = db2.join(temp_gradient2_df_r)   

# 사이클 수 계산
db2 = db2.dropna(axis=0).reset_index(drop=True)
cycle_point_detect = pd.DataFrame([])
        
for i in range(0, len(db2)-1):
    if db2["gradient_1"][i] <= 0 and db2["gradient_1"][i+1] > 0 and db2["gradient_1_2sb"][i] <= 0 and db2["gradient_1_2sf"][i] > 0:
        cycle_point_detect = cycle_point_detect.append([1])
    
    elif db2["gradient_1"][i] < 0 and db2["gradient_1"][i+1] >= 0 and db2["gradient_1_2sb"][i] < 0 and db2["gradient_1_2sf"][i] >= 0:
        cycle_point_detect = cycle_point_detect.append([2])
             
    else :
        cycle_point_detect = cycle_point_detect.append([0])

cycle_point_detect = cycle_point_detect.reset_index(drop=True)
cycle_point_detect.rename(columns={0:"cycle_point"}, inplace = True)
db2 = db2.join(cycle_point_detect)   

point = db2['cycle_point'][db2['cycle_point'] > 0]
point = point.reset_index(drop=False)
db2['cycle'] = np.nan
i = 0
count = 1
while i < len(point)-2: # 사이클 번호 지정
    if point['cycle_point'][i] == 1 and (db2['gradient_1'][point['index'][i]] > 0 or db2['gradient_2'][point['index'][i]] > 0) and abs(db2['5mean_resistance'][point['index'][i]+1] - db2['5mean_resistance'][point['index'][i]]) > 0 and abs(db2['5mean_resistance'][point['index'][i]+2] - db2['5mean_resistance'][point['index'][i]]) > 0.5:        
        db2['cycle'][point['index'][i]:point['index'][i+1]] = count
        count = count +1
    i = i + 1

#pyplot.plot(db2['time'][0:3500], db2['resistance'][0:3500])

db3 = db2.dropna(axis=0)

print(int(db3['cycle'].max()),'사이클 (pre-strain /센서 파단 데이터 포함)')

# 사이클 마다 min max 계산
cycle_max = pd.DataFrame([])
cycle_min = pd.DataFrame([])
for i in range(1, int(db3['cycle'].max())):
    cycle_max = cycle_max.append([db3['resistance'][db3['cycle']==i].max()])
    cycle_min = cycle_min.append([db3['resistance'][db3['cycle']==i].min()])

cycle_max.rename(columns={0:"cycle_max"}, inplace = True)
cycle_min.rename(columns={0:"cycle_min"}, inplace = True)    
cycle_max_min = pd.concat([cycle_max, cycle_min], axis=1)
cycle_max_min_index = pd.DataFrame(np.arange(1, int(db3['cycle'].max()), 1)).reset_index(drop=True)
cycle_max_min_index.rename(columns={0:"cycle_number"}, inplace = True)
cycle_max_min = pd.concat([cycle_max_min.reset_index(drop=True), cycle_max_min_index], axis=1)
cycle_max_min = cycle_max_min.set_index('cycle_number', drop=True)

cycle_max_min['difference'] = cycle_max_min['cycle_max'] - cycle_max_min['cycle_min']

#pyplot.plot(cycle_max_min.index[0:43], cycle_max_min['difference'][0:43])

