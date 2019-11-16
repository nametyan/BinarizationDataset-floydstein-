import numpy as np
import cv2
import glob
from natsort import natsorted
from tqdm import tqdm
import shutil
import os
import sys

size = 256

min_pix = 0#白の表示
max_pix = 255#黒の表示
threshold_pix = 128#閾値(127でもよいが128推奨)
path_big = "C:/pytools/pix2pix-tensorflow/floyd/BIGpicture"
path_cut = "C:/pytools/pix2pix-tensorflow/floyd/cutpicture"
path_dith = "C:/pytools/pix2pix-tensorflow/floyd/ditherd"
path_test = "C:/pytools/pix2pix-tensorflow/floyd/test"
path_train = "C:/pytools/pix2pix-tensorflow/floyd/train"
path_val = "C:/pytools/pix2pix-tensorflow/floyd/val"
load = "/*"
load_big = natsorted(glob.glob(path_big + load))
load_cut = natsorted(glob.glob(path_cut + load))
load_dith = natsorted(glob.glob(path_dith + load))

def removedir():
    shutil.rmtree("C:/pytools/pix2pix-tensorflow/floyd")

def makedir():
    os.makedirs(path_big)
    os.makedirs(path_cut)
    os.makedirs(path_dith)
    os.makedirs(path_test)
    os.makedirs(path_train)
    os.makedirs(path_val)

def clop():
    global datacount
    for fname in tqdm(load_big):
        pass
        img = cv2.imread(fname)#大きい画像の読み取り        
        height, width, channels = img.shape#サイズ取得
        h_num = int(height//size)#height切る回数
        w_num = int(width//size)#width切る回数
        for h in range(h_num):#y軸
            for w in range(w_num):#x軸
                clp = img[size * h : size * (h + 1), size * w : size * (w + 1)]          
                img_array = np.var(clp, axis = 0)
                img_var = np.mean(img_array, axis = 0)
                var_mean = np.mean(img_var)
                if(var_mean > 500):
                    cv2.imwrite(os.path.join(path_cut, "C_" + str(datacount) + ".png"), clp)
                    datacount += 1
                       
def floyd():
    global datacount
    for fname in tqdm(load_cut):
        pass
        img_gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)#グレースケールで読み込み
        img_size = (size, size, 1)#読み込む画像サイズ
        img_floyd = np.zeros(img_size, dtype = np.int16)#ディザリング結果
        pix_err = 0#次のピクセルに渡す誤差
        X1 = 7 / 16
        Y1X_1 = 3 / 16
        Y1 = 5 / 16
        Y1X1 = 1/ 16#各ピクセルの係数(今回はフロイドスタインバーグ)
    
        for y in range(size - 1):#y軸
            for x in range(size - 1):#x軸        
                if((img_gray[y][x] + img_floyd[y][x]) < threshold_pix):#閾値より小さい
                    img_floyd[y][x] = min_pix#白を表示
                    pix_err = img_gray[y][x]#誤差に代入         
                    img_floyd[y][x + 1] += int(X1 * pix_err)                   
                    img_floyd[y + 1][x + 1] += int(Y1X1 * pix_err) 
                    img_floyd[y + 1][x] += int(Y1 * pix_err)
                    img_floyd[y + 1][x - 1] += int(Y1X_1 * pix_err)                      
                else:#閾値より大きい
                    img_floyd[y][x] = max_pix#黒を表示
                    pix_err = img_gray[y][x] - max_pix#差を誤差に代入               
                    img_floyd[y][x + 1] += int(X1 * pix_err)                    
                    img_floyd[y + 1][x + 1] += int(Y1X1 * pix_err)                
                    img_floyd[y + 1][x] += int(Y1 * pix_err)               
                    img_floyd[y + 1][x - 1] += int(Y1X_1 * pix_err)        
        cv2.imwrite(os.path.join(path_dith, "A_" + str(datacount) + ".png"), img_floyd)
        datacount += 1

def concat():
    for i in tqdm(range(len(load_dith))): 
        pass   
        img_A = cv2.imread(path_dith + "/A_" + str(i) + ".png")
        img_B = cv2.imread(path_cut + "/C_" + str(i) + ".png")
        hconcat = cv2.hconcat([img_A, img_B])
        if(i > 0 and i % 6 == 0):
            cv2.imwrite(os.path.join(path_test, str(i) + ".png"), hconcat)
            i += 1
        elif(i > 0 and i % 20 == 0):
            cv2.imwrite(os.path.join(path_val, str(i) + ".png"), hconcat)
            i += 1
        else:
            cv2.imwrite(os.path.join(path_train, str(i) + ".png"), hconcat)
            i += 1
    
if(__name__ == "__main__"):   
    
    key1 = input('input y remove directory ::')
    if(key1 == "y"):
        removedir()
    
    key2 = input('input y made directory C:/pytools/pix2pix-tensorflow/floyd ::')
    if(key2 == "y"):
        makedir()
   
    key3 = input('input y clop ::')
    if(key3 == "y"):
        datacount = 0
        print('clop now')
        clop()
    
    key4 = input('input y floyd ::')
    if(key4 == "y"):
        datacount = 0
        print('Floyd now')
        floyd()

    key5 = input('input y concat ::')
    if(key5 == "y"):    
        print('concat now')
        concat()

       
        

