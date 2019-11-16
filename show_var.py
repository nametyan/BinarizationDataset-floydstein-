import cv2
import numpy as np
import glob
from natsort import natsorted

#Show var data dir
path = glob.glob("C:/pytools/pix2pix-tensorflow/floyd/cutpicture/*")
truepath = natsorted(path)

for fname in truepath:
    img = cv2.imread(fname)   
    img_var_column = np.var(img, axis = 0) 
    var_column_ave = np.mean(img_var_column, axis = 0)
    #Show image var B and G and R
    #print("Image var B,G,R", var_column_ave)
    var = np.mean(var_column_ave)
    print("Image var=",var)
    cv2.namedWindow(str(fname), cv2.WINDOW_NORMAL)
    cv2.imshow(str(fname), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("\n")