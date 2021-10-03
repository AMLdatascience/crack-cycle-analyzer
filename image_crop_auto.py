"""
Created on Wed Sep 15 14:45:36 2021

@author: AML_MASTER
"""

from PIL import Image
from glob import glob

def image_crop_all(path, save_path):
    image_data_list = glob(str(path)+'//*.jpg')
    
    for i in range(len(image_data_list)):
        image1 = Image.open(image_data_list[i])                  
        a = 0.1 # 왼쪽 짜르는 비율
        b = 0.1 # 오른쪽 짜르는 비율
        croppedImage=image1.crop((image1.size[0]*a, 0, image1.size[0]*(1-b-a), image1.size[0]*0.6)) # 이미지 자르기 crop함수 이용 ex. crop(left, up, rigth, down)    
        croppedImage.save(str(save_path) + '//re_' + image_data_list[i][-28:-4] + '.jpg')  #-9 -4는 파일의 이름을 추출하기 위함
    return

path = 'archive\\Positive_crack_sensor'
save_path = 'archive\\Positive_crack_sensor_crop'

image_crop_all(path, save_path)