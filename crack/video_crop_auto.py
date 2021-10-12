# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 15:33:56 2021

@author: AML_MASTER
"""

import cv2 
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("original")
    parser.add_argument("modified")
    return parser.parse_args()

video_file = "archive/video/test2.mp4" # 동영상 파일 경로
file_name = video_file.split('/')[-1].split('.')[0]
save_path = "./archive/Positive_crack_{0}_20/".format(file_name)
os.mkdir(save_path)

cap = cv2.VideoCapture(video_file) # 동영상 캡쳐 객체 생성  ---①
prev = None 
count = 1    
if cap.isOpened():                 # 캡쳐 객체 초기화 확인
    while True:
        ret, img = cap.read()      # 다음 프레임 읽기      --- ②
        if(int(cap.get(1)) % 15 == 0):
            if ret:                     # 프레임 읽기 정상
               # cv2.imshow(video_file, img) # 화면에 표시  --- ③
               # cv2.waitKey(15)            # 25ms 지연(40fps로 가정)   --- ④            
                img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if prev is None: 
                    prev = img2 # 첫 이전 프레임 --- ⑥
                else :
                    score, diff = compare_ssim(prev, img2, full=True)
                    print(f'SSIM: {score:.6f}')
                    if score > 0.9 :
                       cv2.imwrite(save_path + '{0}_{1}_20x.jpg'.format(file_name, str(count).zfill(3)), img[ 130 : 1024,  250 : 1750])
                       count += 1
                    prev = img2
                        
                if cv2.waitKey(1) & 0xFF == ord('q') : break
            
            else:                       # 다음 프레임 읽을 수 없슴,
                break                   # 재생 완료
else:
    print("can't open video.")      # 캡쳐 객체 초기화 실패
cap.release()                       # 캡쳐 자원 반납
cv2.destroyAllWindows()