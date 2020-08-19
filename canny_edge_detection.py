import cv2

img = cv2.imread("images/Lenna.png", cv2.IMREAD_GRAYSCALE)

img_canny = cv2.Canny(img, 50, 150)

cv2.imshow("img_canny", img_canny)

cv2.waitKey()
cv2.destroyAllWindows()