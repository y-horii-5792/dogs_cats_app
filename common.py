#!/usr/bin/python3.11

#
#import
#
import cv2
import numpy as np
#import matplotlib.pyplot as plt
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
#from tensorflow.keras.applications.vgg16 import VGG16
#from tensorflow.keras.models import Model, Sequential
#from tensorflow.keras import optimizers

#import pandas as pd
#from sklearn.model_selection import train_test_split

#import os
#from google.colab import files

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input


#
#犬種、猫種のリスト
#
#dog_cat_breed_list = ['beagle', 
#                  'boxer',
#                  'chihuahua',
#                  'english_cocker_spaniel',
#                  'shiba_inu',
#                  'Abyssinian', 
#                  'Bengal',
#                  'Bombay',
#                  'British_Shorthair',
#                  'Ragdoll']

dog_cat_breed_list = ['beagle', 
                  'boxer']

#NUM_CLASS = 10
NUM_CLASS = 2


IMG_WIDTH  = 200
IMG_HEIGHT = 200
#IMG_WIDTH  = 64
#IMG_HEIGHT = 64



#
#cv2使うかkeras.preprocessing.image使うか
#
#SW_CV2 = 1


print("\n### func def:input_img_process ###\n")
#
#入力画像の処理関数(cv2)
#
def input_img_process_cv2(img_path,
                          img_width,
                          img_height,
                          sw_np_array,
                          sw_dim_add,
                          sw_vgg16_preprocess) :
    img = cv2.imread(img_path)                    #image read
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    #BGR to RGB
    #b, g, r = cv2.split(img)                     #
    #img = cv2.merge([r,g,b])                     #
    img = cv2.resize(img, (img_width,img_height)) #resize
    
    if sw_np_array == 1:
        img = np.array( img )                       #to NumPy   #OK
        #img = np.array( [img] )                     #to NumPy  #OK
    
    if sw_dim_add == 1:
        img = np.expand_dims(img, axis=0)         #VGG16入力のために次元を1つ追加(バッチ次元の追加)   #OK
        #img = img.reshape(1,img_width,img_height,3) #VGG16入力のために次元を1つ追加(バッチ次元の追加)   #OK
        if sw_vgg16_preprocess == 1 :
            img = preprocess_input(img)           #VGG16用の前処理(推奨される処理)
    
    return img
    pass


#
#入力画像の処理関数(keras.preprocessing.image)
#
def input_img_process(img_path,
                      img_width,
                      img_height,
                      sw_np_array,
                      sw_dim_add,
                      sw_vgg16_preprocess) :
    img = load_img(img_path, target_size=(img_width,img_height)) #image read, resize
    
    if sw_np_array == 1:
        #必要
        img = img_to_array(img)               #画像をNumpy配列に変換
    
        ###いらない
        ###img = np.array( [img] )              #to NumPy  #WARNING
        ###img = np.array( img )              #to NumPy  #WARNING
    
    if sw_dim_add == 1:
        ###どっちでもOK
        img = np.expand_dims(img, axis=0)   #VGG16入力のために次元を1つ追加(バッチ次元の追加)  #OK
        #img = img.reshape(1,img_width,img_height,3) #VGG16入力のために次元を1つ追加(バッチ次元の追加)  #OK
        if sw_vgg16_preprocess == 1 :
            img = preprocess_input(img)      #VGG16用の前処理(推奨される処理)
    
    return img
    pass


