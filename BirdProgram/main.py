from django.shortcuts import render

from PIL import Image

import time
import datetime
import glob
import re
import cv2
import threading
import numpy as np
import datetime
import os
#import pygame
from .program_model import AE, make_models

from .views import files

#from media import files



model_paths = ["AEmodel_dark_green20231223.pth", "AEmodel_light_green20231223.pth", "AEmodel_paved_ground20231223.pth", "AEmodel_white20231223.pth"]
models = make_models(model_paths)

#並列化分割数
number_of_threads = 8
#閾値の上限
area_threshold = 1200

#日付を取得する
t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')
now = datetime.datetime.now(JST)
Datetime = '{:%Y%m%d%H%M%S}'.format(now)

#画像分割サイズ
distance = 224

#画像判定用白色画像
red_image = np.zeros((distance, distance, 3), np.uint8)+255

#マルチスレッド開始
#スレッド数でフォルダーを分割
#商
quotient = len(files) // number_of_threads
#余り
remainder = len(files) % number_of_threads

#余り以外のファイル名を分割して１つのリストにする
filelists = [files[quotient * i:quotient * (i + 1)] for i in range(number_of_threads)]
#余りのファイル名のリストを作成
quot_list = files[quotient * number_of_threads:quotient * number_of_threads + remainder]
#合体させて１つのリストにする
filelists.append(quot_list)

def step_1(request):
    context = {
        "message1":"実行中です。",
        "message2":"しばらくお待ちください。"
    }
    return render(request, "BirdProgram/step_1.html", context)


 #画像を分割する関数
def split(FILES):
    #分割後の画像を分割前の画像ごとに格納
    split_images = []
    for i in range(len(FILES)):
        file = FILES[i] #ファイル名
        img = cv2.imread(file) #画像読み込み
        h, w = img.shape[:2] #画像のサイズ
        #分割の始点
        cx = 0
        cy = 0
        for x in range(h//distance):
            for y in range(w//distance):
                #画像の切り取り
                split_img = img[cx:cx+distance, cy:cy+distance]
                #画像の格納
                split_images.append(split_img)
                cy += distance
            cy = 0
            cx += distance
    return split_images
# AutoEncoderに通す関数
def autoencoder(IMAGES, models):
    # AEに通した結果を保存しておくリスト(鳥がいるかいないか)
    AE_judge_result = []
    # AEに通した結果の画像を保存しておくリスト
    AE_image_result = []
    #画像は1枚ずつ
    for idx, image in enumerate(IMAGES):
        #呼び出した関数からは面積と輪郭画像が出力されてくる
        area = AE(image, models)
        #画像はどちらにしても保存
        AE_image_result.append(image)

        #面積がしきい値より小さければ,鳥なしとして格納
        if area < area_threshold:
            AE_judge_result.append(False)
        #面積がしきい値より大きければ,暫定鳥ありとして格納
        else:
            AE_judge_result.append(True)
 
    return AE_judge_result, AE_image_result



 #メインの処理を行う関数
def split_and_autoencoder(fl, al, ai):
    #画像を分割
    images = split(fl)
    # AutoEncoderに通して,判定結果を保存
    judge_list, images_list = autoencoder(images, models)
    al += judge_list
    # AutoEncoderに通して,結果画像を保存
    ai += images_list
    return



