import os
import re
import cv2
import numpy as np
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt

#get file names of frames
frames = os.listdir('images/frames/')
frames.sort(key=lambda f: int(re.sub('\D', '', f)))

# load frames
images = []
for i in tqdm_notebook(frames):
	img = cv2.imread('frames/+i')
	images.append(img)


# specify frame index
idx = 457

'''
# plot frame
plt.figure(figsize=(10,10))
plt.imshow(images[idx][:,:,0], cmap='gray')
plt.show()
'''


# Frame Mask Creation
# create a zero array
stencil = np.zeros_like(images[idx][:,:,0])

# specify coordinates of the polygon
polygon = np.array([50,270], [220,160], [360,160], [480,270])

#fill polygon with ones
cv2.fillConvexPoly(stencil, polygon, 1)

'''
# plot polygon
plt.figure(figsize=(10,10))
plt.imshow(stencil, cmap='gray')
plt.show()
'''


# apply polygon as a mask on the frame
img = cv2.bitwise_and(images[idx][:,:,0], images[idx][:,:,0], mask=stencil)

'''
# plot masked frame
plt.figure(figsize=(10,10))
plt.imshow(img, cmap='gray')
plt.show()
'''


# Image Pre-processing
# 1. Image Thresholding
