import cv2
import numpy
import time

from utils import ImagePreprocessing

if __name__ == "__main__":
	fname = "videos/input2.mp4"
	cap = cv2.VideoCapture(fname)

	prev_time = 0

	while True:
		if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
			cap.open(fname)

		ret, frame = cap.read()

		processed_frame, bird_eye_view = ImagePreprocessing.ImagePreprocessing(frame)

		curr_time = time.time()
		sec = curr_time - prev_time
		prev_time = curr_time
		fps = 1 / sec
		str_fps = "FPS: %0.1f" %fps
		cv2.putText(processed_frame, str_fps, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

		cv2.imshow("VideoFrame", processed_frame)
		cv2.imshow("bird_eye_view", bird_eye_view)

		if cv2.waitKey(30) > 0:
			break

	cap.release()
	cv2.destroyAllWindows()
