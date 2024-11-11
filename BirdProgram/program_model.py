from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
import torch
import numpy as np
import cv2


# 2値化および輪郭抽出を行う関数
def findcontours(img):
    #グレースケールに変換する.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2値化する
    ret, bin_img = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    #クロージング処理のためのカーネル
    kernel = np.ones((3, 3), np.uint8)
    #クロージング処理
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    #輪郭を抽出する.
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #最大面積を保存する
    max_area = 0
    for i, cnt in enumerate(contours):
        #面積
        area = cv2.contourArea(cnt)
        #最大面積を更新
        max_area = max(max_area, area)

    return max_area

#プログラムの最初にモデルをロードする関数
def make_models(model_paths):
    class CustomModel(nn.Module):
        def __init__(self):
            super(CustomModel, self).__init__()
            # Encoderの構築.
            # # nn.Sequential内にはEncoder内で行う一連の処理を記載する.
            # # create_convblockは複数回行う畳み込み処理をまとめた関数.
            # #畳み込み→畳み込み→プーリング→畳み込み・・・・のような動作
            self.Encoder = nn.Sequential(self.create_convblock(3, 16), #256
                                        nn.MaxPool2d((2, 2)),
                                        self.create_convblock(16, 32), #128
                                        nn.MaxPool2d((2, 2)),
                                        self.create_convblock(32, 64), #64
                                        nn.MaxPool2d((2, 2)),
                                        self.create_convblock(64, 128), #32
                                        nn.MaxPool2d((2, 2)),
                                        self.create_convblock(128, 256), #16
                                        nn.MaxPool2d((2, 2)),
                                        self.create_convblock(256, 512), #8
                                        )
            # Decoderの構築.
            # nn.Sequential内にはDecoder内で行う一連の処理を記載する.
            # create_convblockは複数回行う畳み込み処理をまとめた関数.
            # deconvblockは逆畳み込みの一連の処理をまとめた関数
            #逆畳み込み→畳み込み→畳み込み→逆畳み込み→畳み込み・・・・のような動作
            self.Decoder = nn.Sequential(self.create_deconvblock(512, 256), #16
                                        self.create_convblock(256, 256),
                                        self.create_deconvblock(256, 128), #32
                                        self.create_convblock(128, 128),
                                        self.create_deconvblock(128, 64), #64
                                        self.create_convblock(64, 64),
                                        self.create_deconvblock(64, 32), #128
                                        self.create_convblock(32, 32),
                                        self.create_deconvblock(32, 16), #256
                                        self.create_convblock(16, 16),
                                        )
            #出力前の調整用
            self.last_layer = nn.Conv2d(16, 3, 1, 1)

        #畳み込みブロックの中身
        def create_convblock(self, i_fn, o_fn):
            conv_block = nn.Sequential(nn.Conv2d(i_fn, o_fn, 3, 1, 1),
                                    nn.BatchNorm2d(o_fn),
                                    nn.ReLU(),
                                    nn.Conv2d(o_fn, o_fn, 3, 1, 1),
                                    nn.BatchNorm2d(o_fn),
                                    nn.ReLU()
                                    )
            return conv_block
        #逆畳み込みブロックの中身
        def create_deconvblock(self, i_fn , o_fn):
            deconv_block = nn.Sequential(nn.ConvTranspose2d(i_fn, o_fn, kernel_size=2, stride=2),
                                        nn.BatchNorm2d(o_fn),
                                        nn.ReLU(),
                                        )
            return deconv_block
        
        #データの流れを定義
        def forward(self, x):
            x = self.Encoder(x)
            x = self.Decoder(x)
            x = self.last_layer(x)
            
            return x
        
    #作成したモデルを保存
    models = []
    for model_path in model_paths:
        #モデルの枠組みの作成
        model = CustomModel().cuda()
        #作成済みのモデルをロード
        model.load_state_dict(torch.load(model_path))
        #モデルの保存
        models.append(model)
    return models

# AutoEncoderを行う関数
#引数として,分割済みの画像と作成済みのモデルを指定する
def AE(IMG, models):
    #差分画像の合計値の最小値を保存
    min_sum = float('inf')
    #差分画像から抽出された輪郭の面積の最大値を保存
    max_area = float('inf')
    #画像をTensor型にする関数
    prepocess = T.Compose([T.ToTensor()])

    #全てのモデルに対して
    for model in models:
        #モデルをテスト用にして用意
        model.eval()
        #画像をTensor型にする
        img = prepocess(IMG).unsqueeze(0).cuda()
        #テスト用なので勾配計算は無効にする
        with torch.no_grad():
            output = model(img)[0]
        #出力画像を表示できる形式にする
        output = output.cpu().numpy().transpose(1, 2, 0)
        #出力画像の標準化を行う
        output = np.uint8(np.maximum(np.minimum(output*255 , 255), 0))
        #入力画像を表示できる形式にする
        origin = np.uint8(img[0].cpu().numpy().transpose(1, 2, 0)*255)
        #差分画像を作成
        diff = np.uint8(np.abs(output.astype(np.float32)- origin.astype(np.float32)))
        #関数findcontoursで最大面積を計算
        area = findcontours(diff)
        # min_sumを更新
        min_sum = min(min_sum, diff.sum())
        # max_areaを更新
        max_area = min(max_area, area)
    #差分画像の画素値の合計が基準より小さければ輪郭最大面積を返す
    if min_sum < 2400000:
        return max_area
    else:
        return 0