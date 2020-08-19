import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

frames = os.listdir('images/frames/')
frames.sort(key=lambda f: int(re.sub('\D', '', f)))