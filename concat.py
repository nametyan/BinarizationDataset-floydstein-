import cv2
import numpy as np
import glob
import os
from natsort import natsorted

path_A = glob.glob("C:/FSdata/cutpicture/*")
path_B = glob.glob("C:/FSdata/ditherd/*")
path2 = "C:/FSdata/train"
truepath_A = natsorted(path_A)
truepath_B = natsorted(path_B)
concat_num = len(truepath_A)
datacount = 0

for i in range(concat_num):
    img_A = img_B = []
    img_A = cv2.imread("C:/FSdata/cutpicture/A_" + str(i) + ".jpg")
    img_B = cv2.imread("C:/FSdata/ditherd/B_" + str(i) + ".jpg")
    hcon = cv2.hconcat([img_A, img_B])
    cv2.imwrite(os.path.join(path2, str(datacount) + ".jpg"), hcon)
    datacount += 1
    
