import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

os.chdir(r'I:\Xu\Udacity')

# Read in an image, you can also try test1.jpg or test4.jpg
img = mpimg.imread('bridge_shadow.jpg') 

# HLS space
hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
s = hls[:,:,2]

# Gray scale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Sobel X
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
abs_sobelx = np.absolute(sobelx)
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

# Threshold x gradient
thresh = (20,100)
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel > thresh[0]) & (scaled_sobel <= thresh[1])] = 1

# Threshold color channel
s_thresh = (170,255)
s_binary = np.zeros_like(s)
s_binary[(s > s_thresh[0]) & (s <= s_thresh[1])] = 1


# stack each channel
color_binary = np.dstack((np.zeros_like(sxbinary),sxbinary,s_binary))*255

# combine the two binary thresholds
combined_binary = np.zeros_like(s)
combined_binary[(s_binary ==1) | (sxbinary == 1)] =1

#plot 
f, (ax1,ax2) = plt.subplots(1, 2, figsize = (20,10))
ax1.set_title('Stacked thresholds')
ax1.imshow(color_binary)

ax2.set_title('S+Gradient')
ax2.imshow(combined_binary, cmap = 'gray')










