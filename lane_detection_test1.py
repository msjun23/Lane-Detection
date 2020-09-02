import cv2
import numpy as np
from utils import ImagePreprocessing

if __name__ == "__main__":
	## get image
	img = cv2.imread("images/road2.jpg")
	#cv2.imshow("img", img)

	processed_img, bird_eye_view_img = ImagePreprocessing.ImagePreprocessing(img)

	cv2.imshow("processed_img", processed_img)
	cv2.imshow("bird_eye_view_img", bird_eye_view_img)

	cv2.waitKey()
	cv2.destroyAllWindows()
