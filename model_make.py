#!/usr/bin/python3.11

#
#import
#
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers

import pandas as pd
from sklearn.model_selection import train_test_split

import os
from google.colab import files

from keras.preprocessing.image import load_img, img_to_array



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

data  = []
label = []


#
#cv2使うかkeras.preprocessing.image使うか
#
SW_CV2 = 1


print("\n### func def:input_img_process ###\n")
#
#入力画像の処理関数(cv2)
#
def input_img_process_cv2(img_path,
                          img_width,
                          img_height,
                          sw_np_array,
                          sw_dim_add) :
    img = cv2.imread(img_path)                    #image read
    b, g, r = cv2.split(img)                      #r,g,b sort
    img = cv2.merge([r,g,b])
    img = cv2.resize(img, (img_width,img_height)) #resize
    
    
    if sw_np_array == 1:
        #
        #なくてもいい？
        #なくても実行でき、warningも出ない
        #
        img = np.array( img )                       #to np.array   #OK
        #img = np.array( [img] )                     #to np.array  #OK
    
    if sw_dim_add == 1:
        ######img = np.expand_dims(img, axis=0)         #VGG16入力のために次元を1つ追加   #NG
        img = img.reshape(1,img_width,img_height,3) #VGG16入力のために次元を1つ追加   #OK
    
    return img


#
#入力画像の処理関数(keras.preprocessing.image)
#
def input_img_process(img_path,
                      img_width,
                      img_height,
                      sw_np_array,
                      sw_dim_add) :
    img = load_img(img_path, target_size=(img_width,img_height)) #image read
    
    if sw_np_array == 1:
        #必要
        img = img_to_array(img)
    
        ###
        ###いらない
        ###
        ###img = np.array( [img] )              #to np.array  #WARNING
        ###img = np.array( img )              #to np.array  #WARNING
    
    
    if sw_dim_add == 1:
        ###
        ###どっちでもOK
        ###
        #img = np.expand_dims(img, axis=0)  #VGG16入力のために次元を1つ追加  #OK
        img = img.reshape(1,img_width,img_height,3) #VGG16入力のために次元を1つ追加  #OK
    
    return img



#img = preprocess_input(img)                  #VGG16前処理
#img = preprocess_input(img)                  #VGG16前処理
#img = preprocess_input(img)                  #VGG16前処理



#
#入力画像のあるフォルダパス
#
root_path = '/content/drive/MyDrive/Data/images/pet_dataset__part__/'



#
#データ、ラベル準備
#


print("\n### data prepare ###\n")
count = 0

for i in dog_cat_breed_list:
    file_list = os.listdir(root_path + str(i) + '/')
    print("dog_cat_breed_list i: " + i + "\n")
   
    count = 0
    for j in range(len(file_list)): #range(1, 201)

        #print(j + '\n')
        #print(root_path + str(i) + '/' + file_list[j] + '\n')
        count += 1


        img_path = root_path + str(i) + '/' + file_list[j]
        
        if SW_CV2 == 1 :
            x = input_img_process_cv2( img_path, IMG_WIDTH, IMG_HEIGHT, 0, 0 )
        else :
            x = input_img_process(     img_path, IMG_WIDTH, IMG_HEIGHT, 0, 0 )


        #img = cv2.imread(root_path + str(i) + '/' + file_list[j])
        #b, g, r = cv2.split(img)
        #img = cv2.merge([r,g,b])
        ##リサイズ
        #x = cv2.resize(img,(IMG_WIDTH, IMG_HEIGHT))

        #img = load_img(root_path + str(i) +'/' + file_list[j] , target_size=(IMG_WIDTH, IMG_HEIGHT))
        #x = img_to_array(img)

        #if i=='Bengal' and (j>=0 and j<=10): 
        #   plt.imshow(x)
        #   plt.show()


        data.append( x )
        label.append( dog_cat_breed_list.index(i) )
    
    #print(str(i) + "\n")
    #print(str(dog_cat_breed_list.index(i)) + "\n")
    #print(str(dog_cat_breed_list[dog_cat_breed_list.index(i)]) + "\n")
    print("count: " + str(count) + "\n")


#
#X: np.array
#Y: one-hot
#
X = np.array( data )
Y = to_categorical(label)



#
#訓練データと検証データに分ける
#
print("\n### data split ###\n")
X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                    Y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=Y)

##データの並び替え
#np.random.seed(42)
#rand_index = np.random.permutation(np.arange(len(X)))
#X = X[rand_index]
#Y = Y[rand_index]
#
##訓練データと検証データに分ける
#X_train = X[:int(len(X)*0.8)]
#Y_train = Y[:int(len(Y)*0.8)]
#X_test = X[int(len(X)*0.8):]
#Y_test = Y[int(len(Y)*0.8):]



#
#モデル定義、インスタンス（転移学習:VGG16）
#
print("\n### model def ###\n")
input_tensor = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))

#top_model.add(Dense(256, activation='relu'))
#top_model.add(Dense(4, activation='relu'))
#top_model.add(Dense(256, activation='sigmoid'))
#top_model.add(Dropout(0.8))

#top_model.add(Dense(128, activation='relu'))

top_model.add(Dense(NUM_CLASS, activation='softmax'))

model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))


#
#VGG16 重み固定範囲の指定
#
print("\n### weight fix ###\n")
#FIX_LAYER = 15
FIX_LAYER = 19
for layer in model.layers[:FIX_LAYER]:
    layer.trainable = False

#
#コンパイル
#
print("\n### compile ###\n")
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9),
              #optimizer='adadelta',
              metrics=['accuracy'])

print("\n### summary ###\n")
model.summary()

#
#モデル学習
#
print("\n### fit ###\n")
BATCH_SIZE = 32
EPOCHS = 1
history = model.fit(X_train,
                    Y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS)



#
# 精度の評価
#
print("\n### evaluation ###\n")
scores = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss    :', scores[0])
print('Test accuracy:', scores[1])



#
#判定を返す関数
#
print("\n### func def:pred_breed ###\n")

def pred_breed( img ):
    
    #モデル予測
    probability = model.predict( img )
    print('[pred_breed] probability: ' + str(probability))

    #確率の最も高いindex
    pred_index = np.argmax( probability )
    print('[pred_breed] pred_index: ' + str(pred_index))
    
    #最も高い確率
    pred_prob = np.max( probability )
    print('[pred_breed] pred_prob: ' + str(pred_prob))

    #犬種（、猫種）を文字列で返す
    breed = dog_cat_breed_list[pred_index]
    print('[pred_breed] pred: ' + str(breed))

    return breed



#
# 予測(各種類からNUM_OF_TEST枚ずつ)
#
print("\n### prediction ###\n")

NUM_OF_TEST = 4

for i in range( NUM_CLASS ):

    print("i: " + str(i) + "\n")
    print("dog_cat_breed_list[i]: " + dog_cat_breed_list[i] + "\n")
    
    #入力画像ファイルリスト
    file_list = os.listdir(root_path + dog_cat_breed_list[i] + '/' )
    
    for j in range( NUM_OF_TEST ):
        print("j: " + str(j) + "\n")
        
        #入力画像パス
        img_path = root_path + dog_cat_breed_list[i] + '/' + file_list[j]
        print( img_path + '\n')
        
        #入力画像読み込み、処理
        if SW_CV2 == 1 :
            print("CV2 \n")
            img      = input_img_process_cv2( img_path , IMG_WIDTH, IMG_HEIGHT, 1, 1 )
            img_disp = input_img_process_cv2( img_path , IMG_WIDTH, IMG_HEIGHT, 0, 0 ) #表示用
        else :
            print("keras.preprocessing.image \n")
            img      = input_img_process( img_path , IMG_WIDTH, IMG_HEIGHT, 1, 1 )
            img_disp = input_img_process( img_path , IMG_WIDTH, IMG_HEIGHT, 0, 0 ) #表示用
        
        
        #入力画像表示
        plt.imshow( img_disp )
        plt.show()

        #予測
        pred = pred_breed( img )
        print('pred: ' + pred)
        
        #OK, NG表示
        if dog_cat_breed_list[i] == pred :
            print("OK \n")
        else:
            print("NG \n")
        print("------------------------------\n")
    
    





#resultsディレクトリを作成
print("\n### results directory ###\n")
result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

# 重みを保存
model.save(os.path.join(result_dir, 'model.h5'))

files.download( '/content/results/model.h5' ) 


print("\n### file end. ###\n")
######## file end ########
