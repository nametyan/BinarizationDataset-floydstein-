#扱いたいデータサイズに分割する
#分割した際の端数は切り捨てられる
"""
扱う画像のサイズ変更「cut_num」
変換前の画像の格納フォルダ「path」
変換後の画像の格納フォルダ「path2」
"""

import cv2
import numpy as np
import os
import glob
from natsort import natsorted

path = glob.glob("C:/FSdata/BIGpicture/*")#サイズ調整前の画像の格納フォルダ
path2 = "C:/FSdata/cutpicture"#サイズ調整後の画像の格納フォルダ
cut_num = 128#扱いたい画像サイズ[pixel]

makecount = 0#作成した画像の枚数
color = ('b','g','r')
truepath = natsorted(path)

if(__name__ == "__main__"):
    for fname in truepath:
        img = cv2.imread(fname)#大きい画像の読み取り        
        height, width, channels = img.shape#サイズ取得
        h_num = int(height//cut_num)#height切る回数
        w_num = int(width//cut_num)#width切る回数
        for h in range(h_num):#y軸
            for w in range(w_num):#x軸
                clp = img[cut_num * h : cut_num * (h + 1), cut_num * w : cut_num * (w + 1)]#画像を分割できる数とcut_numの積が分割する点になる
                choice_var = 0
                for i,col in enumerate(color):
                    histr = cv2.calcHist([clp],[i],None,[256],[0,256])
                    histarray = np.ravel(histr)#variance = variance(histr)
                    var = np.var(histarray)
                    if(var > 20000):
                        choice_var += 1
                if(choice_var == 0):
                    cv2.imwrite(os.path.join(path2, "A_" + str(makecount) + ".jpg"), clp)
                    makecount += 1
                    
