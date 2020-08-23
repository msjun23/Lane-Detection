import cv2
import numpy as np
#from utils import draw_lines as draw_lines

if __name__ == "__main__":
	## get image
	img = cv2.imread("images/road2.jpg")
	#cv2.imshow("img", img)

	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#cv2.imshow("gray_img", gray_img)


	## rgb -> hsv
	hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

	## masking image (yellow and white)
	lower_yellow = np.array([20,100,100], dtype="uint8")
	upper_yellow = np.array([30,255,255], dtype="uint8")

	mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
	mask_white = cv2.inRange(gray_img, 200, 255)
	mask_y_w = cv2.bitwise_or(mask_yellow, mask_white)
	mask_y_w_img = cv2.bitwise_and(gray_img, mask_y_w)
	#cv2.imshow("mask_y_w", mask_y_w)
	#cv2.imshow("mask_y_w_img", mask_y_w_img)


	## set ROI-Region Of Interest
	mask = np.zeros_like(mask_y_w_img)
	ignore_color = 255

	imshape = mask_y_w_img.shape
	lower_left = [imshape[1]/9,imshape[0]]
	lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
	top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10]
	top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10]
	vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]

	cv2.fillPoly(mask, vertices, ignore_color)

	roi_img = cv2.bitwise_and(mask_y_w_img, mask)
	#cv2.imshow("roi_img", roi_img)


	## apply Gaussian blur to suppress noise in Canny Edge Detection
	kernel_size = (5,5)
	gaussian_img = cv2.GaussianBlur(roi_img, kernel_size, 0)
	#cv2.imshow("gaussian_img", gaussian_img)


	## Canny Edge Detection
	low_threshold = 50
	upper_threshold = 150
	canny_edges = cv2.Canny(gaussian_img, low_threshold, upper_threshold)
	cv2.imshow("canny_edges", canny_edges)


	## Hough Space
	# use cv2.HoughLinesP
	hough_threshold = 10
	minLineLength = 1
	maxLineGap = 20

	lines = cv2.HoughLinesP(canny_edges, 1, np.pi/180, threshold=hough_threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

	for line in lines:
		x1 = line[0][0]
		y1 = line[0][1]
		x2 = line[0][2]
		y2 = line[0][3]

		cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)

	cv2.imshow("lines_img", img)

	cv2.waitKey()
	cv2.destroyAllWindows()
	