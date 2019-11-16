import cv2
import numpy as np
import os
import glob
from natsort import natsorted
from tqdm import tqdm

#before clop dir
path = glob.glob("C:/pytools/pix2pix-tensorflow/floyd/BIGpicture/*")
truepath = natsorted(path)

#after clop dir
path2 = "C:/pytools/pix2pix-tensorflow/floyd/cutpicture"

#clop size[pixel]
cut_num = 256

datacount = 0

def clop(file):
    img = cv2.imread(file)       
    height, width, channels = img.shape
    h_num = int(height//cut_num)
    w_num = int(width//cut_num)
    global datacount
    for h in range(h_num):
        for w in range(w_num):
            clp = img[cut_num * h : cut_num * (h + 1), cut_num * w : cut_num * (w + 1)]           
            img_array = np.var(clp, axis = 0)
            img_var = np.mean(img_array, axis = 0)
            var_min = np.min(img_var)
            if(var_min < 600):#Personal values(change by yourself)
                cv2.imwrite(os.path.join(path2, "C_" + str(datacount) + ".jpg"), clp)
                datacount += 1
                
if(__name__ == "__main__"):
    for fname in tqdm(truepath):
        pass
        clop(fname)
        
