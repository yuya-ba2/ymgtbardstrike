from django.shortcuts import render

#import tkinter
#from tkinter import font
#import cv2
#from PIL import Image , ImageTk
#import re

from django.http import HttpResponse

from .models import Article

from django import forms

from .forms import UploadForm

import os
from django.conf import settings

from django.core.files.storage import FileSystemStorage

files = ()

def GUI_start(request):
    form = UploadForm()
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            files = request.FILES.getlist('file')
            for file in files:
                file_path = os.path.join(settings.MEDIA_ROOT, file.name)         
                with open(file_path, 'wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)
            context = {
                'message1':
                '実行ボタンを押すとプログラムが作動します。',
                'message2':'プログラム作動後しばらく時間がかかるためお待ちください。'
            }

            return render(request, 'BirdProgram/GUI_judge_start.html', context) 
        else:
            context = {
            'message1':'ファイルが入力されていません。',
            'message2':'戻るを押してもう一度ファイルを選択してください。'
            }
            return render(request, 'BirdProgram/GUI_error.html', context)


    context = {
        'message1': 'ただいまより調査を開始します。',
        'message2':'ドローン画像が格納してあるファイルを下記から入力してください。',
        'message3':'<注意事項>',
        'message4':'①途中音声を使用します。音量確認をお願いいたします。',
        'message5':'②可能な限りシステムの×ボタンを押さないようにお願いいたします。',
        'message6':'(システムは全て自動で終了するか、ボタンを押すと終了するようになっています。)',
        'message7':'ファイルを選択しましたら入力を押して下さい。' 
    }

    return render(request, 'BirdProgram/GUI_1.html', context)















#def GUI_error(request):
#    context = {
#        'message1':'プログラムエラーが起きました。',
#        'message2':'戻るを押してもう一度ファイルを選択してください。'
#        }

#    return render(request, 'BirdProgram/GUI_error.html', context)



#def GUI_judge_start(request):




#def upload_file(request):
#    form = UploadForm()
#    if request.method == 'POST':
#        form = UploadForm(request.POST, request.FILES)
#        if form.is_valid():
#            form.save()
#            return HttpResponse("File uploaded successfully.")
#    else:
#        form = UploadForm()
#    return render(request, 'upload.html', {'form': form})

























#def GUI_start(request):

#    folder_name = []

#    root = tkinter.Tk()

#    root.title('プログラムスタート')

#    root.state('zoomed')

#    def button_click():
#        folder_name.append("sample_image")
#        root.destroy()
    
#    sample_button = tkinter.Button(root, text='動作確認', command=button_click, borderwidth=0, relief="solid", width=5, height=1, bg="white", fg="black")

#    sample_button.pack(side="top", anchor="ne")

#    font_nomal = font.Font(family='Helvetica', size=30, weight='bold')

#    label = tkinter.Label(root, text="ただいまより調査を開始します。\nドローンを格納してあるフォルダーの名前を\n下記に入力してください。", font=font_nomal)

#    label.pack(anchor='center')

##    label = tkinter.Label(root, text="ドローンを格納してあるフォルダーの名前を\n下記に入力してください。", font=font_nomal)

#    txt = tkinter.Entry(font=font_nomal, justify=tkinter.CENTER)

#    txt.pack()

#    font1 = font.Font(family='Helvetica', size=15, weight='bold')
#    label = tkinter.Label(root, text="<注意事項>\n①途中音声を使用します。音量確認をお願いいたします。\n②可能な限りシステムの×ボタンを押さないようにお願いいたします。\n(システムは全て自動で終了するか、ボタンを押すと終了するようになっています。)", font=font1, fg="red")
#    label.pack()

#    label = tkinter.Label(root, text="\n入力が完了しましたら、\n下の入力ボタンを押してください。", font=font_nomal)
#    label.pack()

#    def getValue():
#        file = txt.get()
#        folder_name.append("../" + file)
#        root.destroy()
#    button = tkinter.Button(root, text='入力完了', command=getValue, bg='#f0e68c', height=5, width=20)

#    button.pack(side = tkinter.BOTTOM)

#    root.mainloop()

#    return folder_name[0]



# Create your views here.
