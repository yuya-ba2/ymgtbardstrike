from django.shortcuts import render, redirect
from django.conf import settings
import os

import base64
from django.core.paginator import Paginator

#from PIL import Image
#from io import BytesIO

#import json
#import time



def image_view(request):

    media_path = os.path.join(settings.MEDIA_ROOT)
    image_files = [f for f in os.listdir(media_path) if f.endswith(('png', 'jpg', 'jpeg', 'gif'))]

    paginator = Paginator(image_files, 24)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    image_data = []

    for image_file in page_obj:
        image_path = os.path.join(settings.MEDIA_ROOT, image_file)
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
            image_data.append(f'data:image/jpg;base64,{encoded_string}')

    return render(request, 'BirdProgram/image_list.html',{'image_data': image_data, 'page_obj': page_obj})












def bird_ok(request):
    image_data = []    
#    if request.method == 'POST':
#        selected_images = json.loads(request.POST.get('selected_images', '[]'))
#        images = request.POST.getlist('images')

#        save_path = os.path.join(settings.MEDIA_ROOT, 'bird')
#        os.makedirs(save_path, exist_ok = True)

#        for img_src in images:
            # Base64データを解析
#            img_data = img_src.split(',')[1]
#            img_bytes = base64.b64decode(img_data)

            # 新しいファイル名を生成
#            img_name = os.path.basename(img_src)
#            img_save_path = os.path.join(save_path, img_name)

            # 画像を保存
#            with open(img_save_path, 'wb') as f:
#                f.write(img_bytes)

        # 保存後、同じページを表示
#        media_path = os.path.join(settings.MEDIA_ROOT, 'images')
#        image_files = [f for f in os.listdir(media_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

#        paginator = Paginator(image_files, 48)
#        page_number = request.GET.get('page')
#        page_obj = paginator.get_page(page_number)

#        for image_file in page_obj:
#            image_path = os.path.join(media_path, image_file)
#            img = Image.open(image_path)
#            buffer = BytesIO()
#            img.thumbnail((100, 100))
#            img.save(buffer, format='JPEG')
#            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
#            image_data.append(f"data:image/jpeg;base64,{img_str}")

#        return render(request, 'BirdProgram/bird_check.html', {'image_data': image_data, 'page_obj': page_obj, 'success_message': '画像が保存されました！'})

    # GETリクエスト時の処理（画像一覧を表示）
#    return render(request, 'BirdProgram/bird_check.html')

#        for img_data in selected_images:            
#            header, encoded = img_data.split(',', 1)
#            image_data = base64.b64decode(encoded)
#            file_path = os.path.join(settings.MEDIA_ROOT, 'bird', f'image_{int(time.time())}.jpg')
#            
#            with open(file_path, 'wb') as img_file:
#                img_file.write(image_data)
#
#        return redirect('upload_image') 
    
    context = {
        "message1":"選択した画像の位置情報を表示します。"
    }
    
    return render(request, 'BirdProgram/bird_check.html', context)


