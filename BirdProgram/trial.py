from django.shortcuts import render
from django.http import HttpResponse


def program1(request):
    context = {
        'message1':'ただいま実行中です。',
    }
    return render(request, 'BirdProgram/trial.html', context)

def program2(request):
    context = {
        'message1':'作業が終了しました。',
        'message2':'ただいまより判定作業を開始します。',
        'message3':'鳥が確認された箇所をクリックしてください。',
        'message4':'クリックした箇所は赤く表示されます。',
        'message5':'間違えて押してしまった際は',
        'message6':'もう一度クリックすると色が戻ります。',
        'message7':'全て確認しましたら『確認終了』ボタンを押してください。'
    }
    return render(request, 'BirdProgram/trial2.html', context)

#def program3(request):
#    return HttpResponse('Hello')