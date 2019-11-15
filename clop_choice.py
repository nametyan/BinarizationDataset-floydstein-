#扱いたいデータサイズに分割する
#分割した際の端数は切り捨てられる

import cv2
import numpy as np
import os
import glob
from natsort import natsorted
from tqdm import tqdm

path = glob.glob("C:/pytools/pix2pix-tensorflow/floyd/BIGpicture/*")#サイズ調整前の画像の格納フォルダ
path2 = "C:/pytools/pix2pix-tensorflow/floyd/cutpicture"#サイズ調整後の画像の格納フォルダ
cut_num = 256#扱いたい画像サイズ[pixel]

datacount = 0#作成した画像の枚数
truepath = natsorted(path)

def clop(file):
    img = cv2.imread(file)#大きい画像の読み取り        
    height, width, channels = img.shape#サイズ取得
    h_num = int(height//cut_num)#height切る回数
    w_num = int(width//cut_num)#width切る回数
    global datacount
    for h in range(h_num):#y軸
        for w in range(w_num):#x軸
            clp = img[cut_num * h : cut_num * (h + 1), cut_num * w : cut_num * (w + 1)]#画像を分割できる数とcut_numの積が分割する点になる
            choice_var = 0           
            img_array = np.var(clp, axis = 0)
            img_var = np.mean(img_array, axis = 0)
            var_min = np.min(img_var)
            if(var_min < 600):
                cv2.imwrite(os.path.join(path2, "A_" + str(datacount) + ".jpg"), clp)
                datacount += 1
                
if(__name__ == "__main__"):
    for fname in tqdm(truepath):
        pass
        clop(fname)
        
