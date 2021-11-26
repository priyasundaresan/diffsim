import cv2
import numpy as np
import os

if __name__ == '__main__':
    filtered_dir = 'pcls_filtered'
    raw_dir = 'pcls_raw'
    for f in sorted(os.listdir(filtered_dir)):
        img1 = cv2.imread(os.path.join(raw_dir, f))
        img2 = cv2.imread(os.path.join(filtered_dir, f))
        res = np.hstack((img1,img2))
        cv2.imshow('res', res)
        cv2.waitKey(0)
