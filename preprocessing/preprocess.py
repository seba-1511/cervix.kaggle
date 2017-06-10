import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import math
from sklearn import mixture
from sklearn.utils import shuffle
from skimage import measure
import os
from imutils import get_and_crop_image
from imutils import crop_minAreaRect


image=cv2.imread('15.jpg')	
image, rec=get_and_crop_image(image)
image=crop_minAreaRect(image,rec)
image=cv2.resize(image,(256,256))

plt.imshow(image)
plt.show()

	
	
	