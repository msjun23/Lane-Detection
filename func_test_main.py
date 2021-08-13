# This is for testing util scripts

import cv2
import time

from numpy.core.fromnumeric import size
from utils import histogram as hist
from utils import ImagePreprocessing

'''# Test ColorImgCLAHE
img = cv2.imread('images/road00.png')
hist.ColorImgCLAHE(img)'''


if __name__ == "__main__":
    img = cv2.imread('images/yellow_white.png')
    clahe_img = hist.ColorImgCLAHE(img)
    color_detected_img = ImagePreprocessing.MaskImage(clahe_img)
    
    cv2.imshow("color_detected_img", color_detected_img)

    cv2.imshow('hsv', cv2.cvtColor(clahe_img, cv2.COLOR_BGR2HSV))
    #print(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2HSV))
    print(size(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2HSV)))

    fname = "videos/input2.mp4"
    #cap = cv2.VideoCapture(fname)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    prev_time = 0
    
    while True:
        #if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            #cap.open(fname)
            
        ret, frame = cap.read()
        clahe_frame = hist.ColorImgCLAHE(frame)
        color_detected_frame = ImagePreprocessing.MaskImage(clahe_frame)

        # Display frame
        '''curr_time = time.time()
        sec = curr_time - prev_time
        prev_time = curr_time
        fps = 1 / sec
        str_fps = "FPS: %0.1f" %fps
        cv2.putText(color_detected_frame, str_fps, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))'''

        cv2.imshow('frame', frame)
        cv2.imshow('frame hsv', cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        cv2.imshow("VideoFrame", color_detected_frame)

        if cv2.waitKey(30) > 0:
            break

    cap.release()
    cv2.destroyAllWindows()
