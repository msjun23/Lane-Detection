import cv2
import numpy as np

def ImagePreprocessing(img):
	## masking image (yellow and white)
	mask_y_w_img = MaskImage(img)

	## set ROI-Region Of Interest
	## perspective transform
	roi_img, bird_eye_view = SetROI(mask_y_w_img)

	## Canny Edge Detection
	canny_edges = GetCannyEdge(roi_img)

	## Hough Transform
	res = HoughTransform(canny_edges, img)

	return res, bird_eye_view
	

def MaskImage(img):
	## rgb -> hsv
	hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	## rgb -> gray
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	lower_yellow = np.array([20,100,100], dtype="uint8")
	upper_yellow = np.array([30,255,255], dtype="uint8")

	mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
	mask_white = cv2.inRange(gray_img, 200, 255)
	mask_y_w = cv2.bitwise_or(mask_yellow, mask_white)
	mask_y_w_img = cv2.bitwise_and(gray_img, mask_y_w)

	return mask_y_w_img


def SetROI(img):
	mask = np.zeros_like(img)
	ignore_color = (255,0,0)

	imshape = img.shape
	lower_left = [imshape[1]/9, imshape[0]]
	lower_right = [imshape[1]-imshape[1]/9, imshape[0]]
	top_left = [imshape[1]/2-imshape[1]/8, imshape[0]/2+imshape[0]/10]
	top_right = [imshape[1]/2+imshape[1]/8, imshape[0]/2+imshape[0]/10]
	
	## set ROI-Region Of Interest
	vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
	cv2.fillPoly(mask, vertices, ignore_color)
	roi_img = cv2.bitwise_and(img, mask)

	## perspective transform
	points = np.float32([lower_left, top_left, top_right, lower_right])
	bird_eye_view = PerspectiveTransform(roi_img, points)
	
	return roi_img, bird_eye_view


def PerspectiveTransform(img, points, size=(640,480)):
	dst_points = np.float32([(80,480), (80,0), (560,0), (560,480)])

	M = cv2.getPerspectiveTransform(points, dst_points)
	bird_eye_view = cv2.warpPerspective(img, M, size)

	'''
	## morphology
	morph_kernel = np.ones((5,5), np.uint8)
	bird_eye_view = cv2.morphologyEx(bird_eye_view, cv2.MORPH_OPEN, morph_kernel)
	'''

	return bird_eye_view


def GetCannyEdge(img):
	## apply Gaussian blur to suppress noise in Canny Edge Detection
	kernel_size = (5,5)
	gaussian_img = cv2.GaussianBlur(img, kernel_size, 0)

	low_threshold = 50
	upper_threshold = 150
	canny_edges = cv2.Canny(gaussian_img, low_threshold, upper_threshold)

	return canny_edges


def HoughTransform(img, res):
	## use cv2.HoughLinesP
	hough_threshold = 10
	minLineLength = 1
	maxLineGap = 20

	lines = cv2.HoughLinesP(img, 1, np.pi/180, threshold=hough_threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

	DrawLines(res, lines)

	return res


def DrawLines(img, lines, color=[0,0,255], thickness=2):
	for line in lines:
		x1 = line[0][0]
		y1 = line[0][1]
		x2 = line[0][2]
		y2 = line[0][3]

		cv2.line(img, (x1,y1), (x2,y2), color, thickness)
		