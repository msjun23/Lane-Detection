import cv2
import numpy as np

img = cv2.imread("images/building.jpg", cv2.IMREAD_GRAYSCALE)
res = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
edges = cv2.Canny(img, 50, 150)

lines = cv2.HoughLines(edges, 1, np.pi/180, 250)
for line in lines:
	rho, theta = line[0]
	cos_t = np.cos(theta)
	sin_t = np.sin(theta)
	x0 = cos_t * rho
	y0 = sin_t * rho

	a = 1000
	x1 = int(x0 + a*(-sin_t))
	y1 = int(y0 + a*(cos_t))
	x2 = int(x0 - a*(-sin_t))
	y2 = int(y0 - a*(cos_t))

	cv2.line(res, (x1,y1), (x2,y2), (0,0,255), 2)

cv2.imshow("edges", edges)
cv2.imshow("hough transform & lines", res)
cv2.waitKey()
cv2.destroyAllWindows()