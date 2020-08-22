import cv2
import numpy as np

def get_slope(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)

def drawlines(img, lines, color=[255, 0, 0], thickness=6):
    """workflow:
    1) examine each individual line returned by hough & determine if it's in left or right lane by its slope
    because we are working "upside down" with the array, the left lane will have a negative slope and right positive
    2) track extrema
    3) compute averages
    4) solve for b intercept 
    5) use extrema to solve for points
    6) smooth frames and cache
    """
    global cache
    global first_frame
    y_global_min = img.shape[0] #min will be the "highest" y value, or point down the road away from car
    y_max = img.shape[0]
    l_slope, r_slope = [],[]
    l_lane,r_lane = [],[]
    det_slope = 0.4
    α =0.2 
    #i got this alpha value off of the forums for the weighting between frames.
    #i understand what it does, but i dont understand where it comes from
    #much like some of the parameters in the hough function
    
    for line in lines:
        #1
        for x1,y1,x2,y2 in line:
            slope = get_slope(x1,y1,x2,y2)
            if slope > det_slope:
                r_slope.append(slope)
                r_lane.append(line)
            elif slope < -det_slope:
                l_slope.append(slope)
                l_lane.append(line)
        #2
        y_global_min = min(y1,y2,y_global_min)
    
    # to prevent errors in challenge video from dividing by zero
    if((len(l_lane) == 0) or (len(r_lane) == 0)):
        print ('no lane detected')
        return 1
        
    #3
    l_slope_mean = np.mean(l_slope,axis =0)
    r_slope_mean = np.mean(r_slope,axis =0)
    l_mean = np.mean(np.array(l_lane),axis=0)
    r_mean = np.mean(np.array(r_lane),axis=0)
    
    if ((r_slope_mean == 0) or (l_slope_mean == 0 )):
        print('dividing by zero')
        return 1
    
   
    
    #4, y=mx+b -> b = y -mx
    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])
    
    #5, using y-extrema (#2), b intercept (#4), and slope (#3) solve for x using y=mx+b
    # x = (y-b)/m
    # these 4 points are our two lines that we will pass to the draw function
    l_x1 = int((y_global_min - l_b)/l_slope_mean) 
    l_x2 = int((y_max - l_b)/l_slope_mean)   
    r_x1 = int((y_global_min - r_b)/r_slope_mean)
    r_x2 = int((y_max - r_b)/r_slope_mean)
    
    #6
    if l_x1 > r_x1:
        l_x1 = int((l_x1+r_x1)/2)
        r_x1 = l_x1
        l_y1 = int((l_slope_mean * l_x1 ) + l_b)
        r_y1 = int((r_slope_mean * r_x1 ) + r_b)
        l_y2 = int((l_slope_mean * l_x2 ) + l_b)
        r_y2 = int((r_slope_mean * r_x2 ) + r_b)
    else:
        l_y1 = y_global_min
        l_y2 = y_max
        r_y1 = y_global_min
        r_y2 = y_max
      
    current_frame = np.array([l_x1,l_y1,l_x2,l_y2,r_x1,r_y1,r_x2,r_y2],dtype ="float32")
    
    if first_frame == 1:
        next_frame = current_frame        
        first_frame = 0        
    else :
        prev_frame = cache
        next_frame = (1-α)*prev_frame+α*current_frame
             
    cv2.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]),int(next_frame[3])), color, thickness)
    cv2.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]),int(next_frame[7])), color, thickness)
    
    cache = next_frame


if __name__ == "__main__":
	## get image
	img = cv2.imread("images/road2.jpg")
	cv2.imshow("img", img)

	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.imshow("gray_img", gray_img)


	## rgb -> hsv
	hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	lower_yellow = np.array([20,100,100], dtype="uint8")
	upper_yellow = np.array([30,255,255], dtype="uint8")

	## masking image (yellow and white)
	mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
	mask_white = cv2.inRange(gray_img, 200, 255)
	mask_y_w = cv2.bitwise_or(mask_yellow, mask_white)
	mask_y_w_img = cv2.bitwise_and(gray_img, mask_y_w)
	cv2.imshow("mask_y_w", mask_y_w)
	cv2.imshow("mask_y_w_img", mask_y_w_img)


	## apply Gaussian blur to suppress noise in Canny Edge Detection
	kernel_size = (5,5)
	gaussian_img = cv2.GaussianBlur(mask_y_w_img, kernel_size, 0)
	cv2.imshow("gaussian_img", gaussian_img)


	## Canny Edge Detection
	low_threshold = 50
	upper_threshold = 150
	canny_edges = cv2.Canny(gaussian_img, low_threshold, upper_threshold)
	cv2.imshow("canny_edges", canny_edges)


	## set ROI-Region Of Interest
	mask = np.zeros_like(canny_edges)
	ignore_color = 255

	imshape = img.shape
	lower_left = [imshape[1]/9,imshape[0]]
	lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
	top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10]
	top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10]
	vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]

	cv2.fillPoly(mask, vertices, ignore_color)

	roi_img = cv2.bitwise_and(canny_edges, mask)
	cv2.imshow("roi_img", roi_img)


	## Hough Space
	rho = 4
	theta = np.pi/180
	hough_threshold = 30
	min_line_len = 100
	max_line_gap = 10

	lines = cv2.HoughLinesP(roi_img, rho, theta, hough_threshold, np.array([]), min_line_len, max_line_gap)
	line_img = np.zeros((roi_img.shape[0], roi_img.shape[1], 3), dtype=np.uint8)
	drawlines(line_img, lines)
	
	res = cv2.addWeighted(img, 0.8, line_img, 1, 0.)
	cv2.imshow("res", res)


	cv2.waitKey()
	cv2.destroyAllWindows()