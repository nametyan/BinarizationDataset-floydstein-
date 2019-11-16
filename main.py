import numpy as np
import cv2
import glob
from natsort import natsorted
from tqdm import tqdm
import shutil
import os
import sys

#Make dataset folder
dir_tree = "C:/pytools/pix2pix-tensorflow/floyd"
#Use datasize[Pixcel]　
size = 256
#Remove data with little information
clop_choice = 500

min_pix = 0
max_pix = 255
threshold_pix = 128
path_big = dir_tree + "/BIGpicture"
path_cut = dir_tree + "/cutpicture"
path_dith = dir_tree + "/ditherd"
path_test = dir_tree + "/test"
path_train = dir_tree + "/train"
path_val = dir_tree + "/val"
glob_big = path_big + "/*"
glob_cut = path_cut + "/*"
glob_dith = path_dith + "/*"

def removedir():
    try:
        shutil.rmtree(dir_tree)
    except OSError as e:
        pass

def makedir():
    try:
        os.makedirs(path_big)
        os.makedirs(path_cut)
        os.makedirs(path_dith)
        os.makedirs(path_test)
        os.makedirs(path_train)
        os.makedirs(path_val)
    except OSError as e:
        pass

def clop():
    global datacount
    for fname in tqdm(load_big):
        pass
        img = cv2.imread(fname)      
        height, width, channels = img.shape
        h_num = int(height//size)
        w_num = int(width//size)
        for h in range(h_num):
            for w in range(w_num):
                clp = img[size * h : size * (h + 1), size * w : size * (w + 1)]          
                img_array = np.var(clp, axis = 0)
                img_var = np.mean(img_array, axis = 0)
                var_mean = np.mean(img_var)
                if(var_mean > clop_choice):
                    cv2.imwrite(os.path.join(path_cut, "C_" + str(datacount) + ".png"), clp)
                    datacount += 1
                       
def floyd():
    global datacount
    for fname in tqdm(load_cut):
        pass
        img_gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        img_size = (size, size, 1)
        img_floyd = np.zeros(img_size, dtype = np.int16)
        X1 = 7 / 16
        Y1X_1 = 3 / 16
        Y1 = 5 / 16
        Y1X1 = 1/ 16
    
        for y in range(size - 1):
            for x in range(size - 1):        
                if((img_gray[y][x] + img_floyd[y][x]) < threshold_pix):
                    img_floyd[y][x] = min_pix
                    pix_err = img_gray[y][x]       
                    img_floyd[y][x + 1] += int(X1 * pix_err)                   
                    img_floyd[y + 1][x + 1] += int(Y1X1 * pix_err) 
                    img_floyd[y + 1][x] += int(Y1 * pix_err)
                    img_floyd[y + 1][x - 1] += int(Y1X_1 * pix_err)                      
                else:
                    img_floyd[y][x] = max_pix
                    pix_err = img_gray[y][x] - max_pix              
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
        if(len(load_dith) > 1000):
            if(i > 0 and i % 11 == 0):
                cv2.imwrite(os.path.join(path_test, str(i) + ".png"), hconcat)
            elif(i > 0 and (i % int(len(load_dith)//100) == 0)):
                cv2.imwrite(os.path.join(path_val, str(i) + ".png"), hconcat)
            else:
                cv2.imwrite(os.path.join(path_train, str(i) + ".png"), hconcat)
        else:
            if(i > 0 and i % 10 == 0):
                cv2.imwrite(os.path.join(path_test, str(i) + ".png"), hconcat)
            elif(i > 0 and i % 19 == 0):
                cv2.imwrite(os.path.join(path_val, str(i) + ".png"), hconcat)
            else:
                cv2.imwrite(os.path.join(path_train, str(i) + ".png"), hconcat)

if(__name__ == "__main__"):   
    
    print("Make dataset for " + str(dir_tree) + "\n")

    print("Do you want remove directory?\n")
    key1 = input('[y]Remove directory ' + str(dir_tree) + '\n[n]Close\n[none]Skip\n')
    if(key1 == "y"):
        removedir()
        print("Removed directory!\n")
    elif(key1 == "n"):
        sys.exit()
    else:
        print("Skip remove directory!\n")
    
    print("Do you want make directory?\n")
    key2 = input('[y]Make directory ' + str(dir_tree) + '\n[n]Close\n[none]Skip\n')
    if(key2 == "y"):
        makedir()
        print("Maked directory!\n")
    elif(key2 == "n"):
        sys.exit()
    else:
        print("Skip make directory!\n")

    print("Data into " +str(path_big) + " already?\n")
    print("Do you want clop data for " + str(path_cut) + " ?\n")
    key3 = input('[y]Clop data\n[n]Close\n[none]Skip\n[c]Not check\n')
    if(key3 == "y" or key3 == "c"):
        datacount = 0
        load_big = natsorted(glob.glob(glob_big))
        print('Clop now!')
        clop()
        print("\n")
    elif(key3 == "n"):
        sys.exit()
    else:
        print("Skip clop data!\n")
    
    if(key3 == "c"):
        datacount = 0
        load_cut = natsorted(glob.glob(glob_cut))
        print('Floyd now!')
        floyd()
        key4 = key3
    else:
        print("Do you want binarization data for " + str(path_dith) + " ?\n")
        key4 = input('[y]Binarization data\n[n]Close\n[none]Skip\n[c]Not check\n')
        if(key4 == "y" or key4 == "c"):
            datacount = 0
            load_cut = natsorted(glob.glob(glob_cut))
            print('Floyd now!')
            floyd()
            print("\n")
        elif(key4 == "n"):
            sys.exit()
        else:
            print("Skip Binarization data!\n")
    
    if(key4 == "c"):
        load_dith = natsorted(glob.glob(glob_dith))
        print('Concat now!')
        concat()
    else:
        print("Do you want concat data?\n")
        key5 = input('[y]Concat data\n[none]Close\n')
        if(key5 == "y"): 
            load_dith = natsorted(glob.glob(glob_dith))   
            print('Concat now!')
            concat()
        else:
            print("Close!")
            sys.exit()


