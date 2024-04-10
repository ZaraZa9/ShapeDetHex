import cv2
import numpy
import math
import time
import os

print(os.path.isfile('image0.jpg'))
sample = cv2.imread('image0.jpg')
cv2.imshow("Hello",sample)
cv2.waitKey(0)