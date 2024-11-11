import glob
import time
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch
from PIL import Image

#学習用画像集(既に分割済み)
train_images = ["train_image/dark_green", "train_image/light_green", "train_image/around_wind", "train_image/white"]
#作成したモデルの保存先の名前
model_names = ["dark_green20231223", "light_green20231223", "around_wind20231223", "white20231223"]
for i in range(len(model_names)):
#画像集のパス
    dir_path = train_images[i]
#モデルの名前
    model_path = "AEmodel_" + model_names[i] + ".pth"

#学習用画像
    study_data = glob.glob(f"{dir_path}/*")

#画像を読み込んでTensor型にするクラス
    class Custom_Dataset(Dataset):
        def __init__(self, img_list):
            self.img_list = img_list
            self.prepocess = T.Compose([T.ToTensor()])
        def __getitem__(self, idx):
            img = Image.open(self.img_list[idx])
            img = self.prepocess(img)
            return img
        def __len__(self):
            return len(self.img_list)

#データを学習用・評価用に8:2へ分割
    train_list = study_data[:int(len(study_data)*0.8)]
    val_list = study_data[int(len(study_data)*0.8):]
    train_dataset = Custom_Dataset(train_list)
    val_dataset = Custom_Dataset(val_list)
    train_loader = DataLoader(train_dataset, batch_size = 32)
    val_loader = DataLoader(val_dataset, batch_size = 32)

#クラスCustomModelに関しては,
#「プログラムモデルロードファイル」で解説済みなため省略する
    class CustomModel(nn.Module):
        def __init__(self):
            super(CustomModel, self).__init__()
            self.Encoder = nn.Sequential(self.create_convblock(3, 16),
                                        nn.MaxPool2d((2, 2)),
                                        self.create_convblock(16, 32),
                                        nn.MaxPool2d((2, 2)),self.create_convblock(32, 64),
                                        nn.MaxPool2d((2, 2)),
                                        self.create_convblock(64, 128),
                                        nn.MaxPool2d((2, 2)),
                                        self.create_convblock(128, 256),
                                        nn.MaxPool2d((2, 2)),
                                        self.create_convblock(256, 512),
                                        )
            self.Decoder = nn.Sequential(self.create_deconvblock(512, 256),
                                        self.create_convblock(256, 256),
                                        self.create_deconvblock(256, 128),
                                        self.create_convblock(128, 128),
                                        self.create_deconvblock(128, 64),
                                        self.create_convblock(64, 64),
                                        self.create_deconvblock(64, 32),
                                        self.create_convblock(32, 32),
                                        self.create_deconvblock(32, 16),
                                        self.create_convblock(16, 16),
                                        )
            self.last_layer = nn.Conv2d(16, 3, 1, 1)

        def create_convblock(self, i_fn, o_fn):
            conv_block = nn.Sequential(nn.Conv2d(i_fn, o_fn, 3, 1, 1),
                                    nn.BatchNorm2d(o_fn),
                                    nn.ReLU(),
                                    nn.Conv2d(o_fn, o_fn, 3, 1, 1),
                                    nn.BatchNorm2d(o_fn),
                                    nn.ReLU()
                                    )
            return conv_block

        def create_deconvblock(self, i_fn , o_fn):
            deconv_block = nn.Sequential(nn.ConvTranspose2d(i_fn, o_fn, kernel_size=2, stride=2),
                                        nn.BatchNorm2d(o_fn),
                                        nn.ReLU(),
                                        )
            return deconv_block

        def forward(self, x):
            x = self.Encoder(x)
            x = self.Decoder(x)
            x = self.last_layer(x)
            return x


#最大エポック(これ以上は学習しない)
    epoch_num = 5000

# CUDAを設定
    device = 'cuda'
#誤差の最小値を保存
    best_loss = None
#モデルを初期化
    model = CustomModel().to(device)
# limit_epoch回更新されなければ学習を終了する
    limit_epoch = 100

#モデルの最適化手法
    optimizer = optim.Adam(model.parameters())
#誤差関数
    criterion = nn.MSELoss()
    loss_list = {"train":[], "val":[]}

    counter = 0
    for e in range(epoch_num):
        total_loss = 0
#モデルを学習用に設定
        model.train()
        with tqdm(train_loader) as pbar:
            for itr , data in enumerate(pbar):
#モデル内のパラメータの勾配を初期化
                optimizer.zero_grad()
                data = data.to(device)
#モデルに通して出力画像を生成
                output = model(data)
#入力画像と出力画像から誤差関数で誤差を生成
                loss = criterion(output , data)
#これまでのlossを反映
                total_loss += loss.detach().item()
                pbar.set_description(f"[train] Epoch {e+1:03}/{epoch_num:03} Itr {itr+1:02}/{len(pbar):02} Loss {total_loss/(itr+1):.3f} Counter {counter}")
#誤差逆伝播
                loss.backward()
#学習の反映
                optimizer.step()

        loss_list["train"].append(total_loss)
        total_loss = 0
#モデルを検証用に設定
        model.eval()
        with tqdm(val_loader) as pbar:
            for itr , data in enumerate(pbar):
                data = data.to(device)
                with torch.no_grad():
                    output = model(data)
                loss = criterion(output , data)
                total_loss += loss.detach().item()
                pbar.set_description(f"[ val ] Epoch {e+1:03}/{epoch_num:03} Itr {itr+1:02}/{len(pbar):02} Loss {total_loss/(itr+1):.3f}")

#ロスがまだ設定されていなかったり,これ以上学習しても意味がなくなったら
#モデルを保存する
        if best_loss is None or best_loss > total_loss/(itr+1):
            if best_loss is not None:
                print(f"update best_loss {best_loss:.6f} to {total_loss/(itr+1):.6f}")
            best_loss = total_loss/(itr+1)
            torch.save(model.state_dict(), model_path)
            counter = 0
#ロスが更新され続けていたら,
#パラメータの更新を続ける
        else:
            counter += 1
#エポックを超えたらそこで終了する
            if limit_epoch <= counter:
                break
        loss_list["val"].append(total_loss)