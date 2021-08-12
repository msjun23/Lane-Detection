import cv2
import numpy as np
from matplotlib import pyplot as plt

# Contrast Limited Adaptive Histogram Equalization
# for Gray & Color image
def GrayImgCLAHE(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img2 = clahe.apply(img)

    # For visualization
    '''res = cv2.resize(img, (400, 400))
    res2 = cv2.resize(img2, (400, 400))
    res_gray = np.hstack((res, res2))
    cv2.imshow('res_gray', res_gray)
    cv2.waitKey()
    cv2.destroyAllWindows()'''

    return img2

def ColorImgCLAHE(img):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_channels = cv2.split(lab_img)
    
    # lab_channels[0] is lightness channel
    # Do CLAHE for lightness channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab_channels[0] = clahe.apply(lab_channels[0])
    
    lab_img = cv2.merge(lab_channels)
    clahe_bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

    # For visualization
    '''cv2.imshow('clahe_bgr_img', clahe_bgr_img)
    cv2.waitKey()
    cv2.destroyAllWindows()'''

    return clahe_bgr_img
