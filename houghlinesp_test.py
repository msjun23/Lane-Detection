import cv2
import numpy as np

cnt = 0

img = cv2.imread("images/building.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150)

threshold = 200
minLineLength = 160
maxLineGap = 10

lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
for line in lines:
	x1 = line[0][0]
	y1 = line[0][1]
	x2 = line[0][2]
	y2 = line[0][3]
	cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)
	cnt += 1

print(cnt)
cv2.imshow("img", img)
#cv2.imshow("gray", gray)
cv2.imshow("edges", edges)

cv2.waitKey()
cv2.destroyAllWindows()