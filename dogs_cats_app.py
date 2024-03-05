#!/usr/bin/python3.11
import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image

import numpy as np
#import cv2

classes = ["beagle", 
           "boxer",
           "chihuahua",
           "english_cocker_spaniel",
           "shiba_inu",
           "Abyssinian", 
           "Bengal",
           "Bombay",
           "British_Shorthair",
           "Ragdoll"]
image_size = 200


UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('./model.h5')#学習済みモデルをロード


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #受け取った画像を読み込み、np形式に変換
            img = image.load_img(filepath, target_size=(image_size,image_size))
            img = image.img_to_array(img)
            #img = np.array([img])
            img = np.expand_dims(img, axis=0)
            
            #img = cv2.imread( filepath )
            #b, g, r = cv2.split(img)
            #img = cv2.merge([r,g,b])
            #img = cv2.resize(img,(image_size, image_size))
            #img = img.reshape(1,image_size,image_size,3)
            
            #データをモデルに渡して予測
            #確率の配列が戻り値
            result = model.predict(img)
            
            pred_index = np.argmax( result )
            pred_prob  = result[pred_index]
            pred_breed = classes[pred_index]
            
            pred_answer="エラー"
            if pred_prob <= 0.7 :
                pred_answer = "すみません、判定できません"
            elif pred_prob <= 0.9 :
                pred_answer =  pred_breed  + "かもしれません"
            elif pred_prob <= 0.99 :
                pred_answer =  pred_breed  + "の可能性が高いです"
            else :
                pred_answer =  pred_breed  + "です"

            return render_template("index.html",answer=pred_answer)

    return render_template("index.html",answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)
