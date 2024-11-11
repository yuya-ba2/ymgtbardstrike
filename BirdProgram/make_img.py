import numpy as np
import cv2
import os
import tempfile

from django.http import HttpResponse
import base64
from django.shortcuts import render

def create_random_images(num_images, image_size):
    """
    ランダムな疑似画像を作成し、それをcv2で読み込んだリストを作成する関数

    :param num_images: 作成する画像の枚数
    :param image_size: 画像のサイズ (例: (height, width, channels))
    :return: cv2で読み込んだ画像のリスト
    """
    image_list = []
    for _ in range(num_images):
        # ランダムな画像を生成
        random_image = np.random.randint(0, 256, image_size, dtype=np.uint8)
        
        # 一時ファイルに保存
#        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
#            cv2.imwrite(temp_file.name, random_image)
#            temp_filename = temp_file.name
        
        # 画像をcv2で読み込んでリストに追加
#        img = cv2.imread(temp_filename)
#        image_list.append(img)
        
        # 一時ファイルを削除
#        os.remove(temp_filename)

         # 画像をエンコード
        _, buffer = cv2.imencode('.png', random_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Data URIスキームを使って画像データを作成
        image_list.append(f"data:image/png;base64,{img_base64}")
    
    return image_list



# 使用例
def create_random_images_trial(request):
    num_images = 20  # 生成する画像の枚数
    image_size = (224, 224, 3)  # 画像のサイズ (高さ, 幅, チャンネル数)
    images = create_random_images(num_images, image_size)
    
    return render(request, 'BirdProgram/random_images.html', {'images': images})