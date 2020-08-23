import cv2
import numpy
from utils import ImagePreprocessing as ImagePreprocessing

if __name__ == "__main__":
	fname = "videos/input2.mp4"
	cap = cv2.VideoCapture(fname)

	while True:
		if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
			cap.open(fname)

		ret, frame = cap.read()
		processed_frame = ImagePreprocessing.ImagePreprocessing(frame)
		cv2.imshow("VideoFrame", processed_frame)

		if cv2.waitKey(30) > 0:
			break

	cap.release()
	cv2.destroyAllWindows()
