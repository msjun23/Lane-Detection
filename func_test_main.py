# This is for testing util scripts

import cv2
import time
from utils import histogram as hist
from utils import ImagePreprocessing

'''# Test ColorImgCLAHE
img = cv2.imread('images/road00.png')
hist.ColorImgCLAHE(img)'''


if __name__ == "__main__":
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
        yellow_detected_img = ImagePreprocessing.MaskImage(clahe_frame)

        curr_time = time.time()
        sec = curr_time - prev_time
        prev_time = curr_time
        fps = 1 / sec
        str_fps = "FPS: %0.1f" %fps
        cv2.putText(yellow_detected_img, str_fps, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

        cv2.imshow("VideoFrame", yellow_detected_img)

        if cv2.waitKey(30) > 0:
            break

    cap.release()
    cv2.destroyAllWindows()
