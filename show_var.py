import cv2
import numpy as np
import os
import glob
from natsort import natsorted
from statistics import variance

path = glob.glob("C:/pytools/pix2pix-tensorflow/floyd/cutpicture/*")
truepath = natsorted(path)

for fname in truepath:
    img = cv2.imread(fname)   
    img_array = np.var(img, axis = 0)
    img_var = np.mean(img_array, axis = 0)
    var = np.mean(img_var)
    print("var=",var)
    cv2.namedWindow(str(fname), cv2.WINDOW_NORMAL)
    cv2.imshow(str(fname), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("\n")