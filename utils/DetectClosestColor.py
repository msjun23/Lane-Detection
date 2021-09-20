import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import shape

delta_h = 30
delta_sv = 75

def DetectYellow(img):
    hsv_yellow = np.array([30, 255, 255])
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    dist_img = np.sum((hsv_img - hsv_yellow)**2, axis=2)**(1/2)
    pixel_min = np.min(dist_img)
    pixel_max = np.max(dist_img)

    # Pixel Normalization: 0~255
    dist_img = ((dist_img - pixel_min) / (pixel_max - pixel_min)) * 255

    yellow_mask = cv2.inRange(dist_img, np.array([0]), np.array([40]))

    return yellow_mask

def DetectWhite(img):
    hsv_white = np.array([0, 0, 255])
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    dist_img = np.sum((hsv_img - hsv_white)**2, axis=2)**(1/2)
    pixel_min = np.min(dist_img)
    pixel_max = np.max(dist_img)

    # Pixel Normalization: 0~255
    dist_img = ((dist_img - pixel_min) / (pixel_max - pixel_min)) * 255

    white_mask = cv2.inRange(dist_img, np.array([0]), np.array([20]))

    return white_mask

def DetectYellowWhite(img):
    yellow_mask = DetectYellow(img)
    white_mask = DetectWhite(img)

    return cv2.bitwise_or(yellow_mask, white_mask)
