import numpy as np
import cv2

#image size
size = 256
#black pixcel
min_pix = 0
#white pixcel
max_pix = 255
#filename
fname = "in.png"
#load gray scale
img_gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
#threshold 
threshold_pix = 255//2
#image size(taple)
img_size = (size, size, 1)

def floyd():  
    img_floyd = np.zeros(img_size, dtype = np.float16)
    X1 = 7 / 16
    X2 = 3 / 16
    X3 = 5 / 16
    X4 = 1/ 16
    
    for h in range(size):
        for w in range(size):        
            #show white pixcel
            if((img_gray[h][w] + img_floyd[h][w]) > threshold_pix):
                pix_err = img_gray[h][w] + img_floyd[h][w] - max_pix
                img_floyd[h][w] = max_pix                 
            #show black pixcel
            else:
                pix_err = img_gray[h][w] + img_floyd[h][w]
                img_floyd[h][w] = min_pix
            #error propagation                
            if(h == size - 1):
                if(w < size - 1):
                    img_floyd[h][w + 1] += (X1 * pix_err)
            elif(w == size - 1):
                img_floyd[h + 1][w - 1] += (X2 * pix_err)
                img_floyd[h + 1][w] += (X3 * pix_err)
            else:
                img_floyd[h][w + 1] += (X1 * pix_err)                   
                img_floyd[h + 1][w - 1] += (X2 * pix_err)
                img_floyd[h + 1][w] += (X3 * pix_err)
                img_floyd[h + 1][w + 1] += (X4 * pix_err)
    return img_floyd.astype(np.uint8)

if(__name__ == '__main__'):
    img_bin = floyd()
    cv2.imwrite('out.png', img_bin)
