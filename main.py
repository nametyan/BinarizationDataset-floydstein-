#指定の画像をフロイドスタインバーグ法によって二値化する
"""
扱う画像のサイズ変更「s」
変換前の画像の格納フォルダ「path」
変換後の画像の格納フォルダ「path2」
"""

import cv2
import numpy as np
import glob
import os

s = 128#読み込む画像の縦横の長さ[pixel]
path = glob.glob("C:/FSdata/cutpicture/*")#変換前の画像の格納フォルダ
path2 = "C:/FSdata/ditherd"#変換後の画像の格納フォルダ

min_pix = 0#白の表示
max_pix = 255#黒の表示
threshold_pix = 128#閾値(127でもよいが128推奨)
datacount = 0#作成した画像の枚数

def floyd(file):
    img_gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)#グレースケールで読み込み
    img_size = (s, s, 1)#読み込む画像サイズ
    img_floyd = np.zeros(img_size, dtype = np.uint8)#ディザリング結果
    img_err = np.zeros(img_size, dtype = np.float16)#誤差拡散用配列
    pix_err = 0#次のピクセルに渡す誤差
    X1 = 7 / 16
    Y1X_1 = 3 / 16
    Y1 = 5 / 16
    Y1X1 = 1/ 16#各ピクセルの係数(今回はフロイドスタインバーグ)
    
    for y in range(s - 1):#y軸
        for x in range(s - 1):#x軸            
            if((img_gray[y][x] + int(img_err[y][x])) < threshold_pix):#閾値より小さい
                img_floyd[y][x] = min_pix#白を表示
                pix_err = img_gray[y][x]#誤差に代入         
                img_err[y][x + 1] += int(X1 * pix_err)                   
                img_err[y + 1][x + 1] += int(Y1X1 * pix_err) 
                img_err[y + 1][x] += int(Y1 * pix_err)
                img_err[y + 1][x - 1] += int(Y1X_1 * pix_err)                      
            else:#閾値より大きい
                img_floyd[y][x] = max_pix#黒を表示
                pix_err = img_gray[y][x] - max_pix#差を誤差に代入               
                img_err[y][x + 1] += int(X1 * pix_err)                    
                img_err[y + 1][x + 1] += int(Y1X1 * pix_err)                
                img_err[y + 1][x] += int(Y1 * pix_err)               
                img_err[y + 1][x - 1] += int(Y1X_1 * pix_err)
    return img_floyd
   
if(__name__ == '__main__'):
    for fname in path:        
        output = floyd(fname)
        cv2.imwrite(os.path.join(path2, str(datacount) + ".jpg"), output)#path2に画像を保存
        datacount += 1
