def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F,1,0,ksize = sobel_kernel))
    scaled_sobel = np.uint8(255*sobelx/np.max(sobelx))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def rgb_thresh(img, channel = 'r', thresh = (0,255)):
    if channel == 'g':
        ch = 1
    elif channel == 'b':
        ch = 0
    else:
        ch = 2
    channel_img = img[:,:,ch]
    rgb_binary = np.zeros_like(channel_img)
    rgb_binary[(channel_img >= thresh[0]) & (channel_img <= thresh[1])] = 1
    return rgb_binary

def hls_thresh(img, channel = 's',thresh=(0, 255)):
    hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    if channel == 'h':
        ch = 0
    elif channel == 'l':
        ch = 1
    else:
        ch = 2
    channel_img = hls[:,:,ch]
    hls_binary = np.zeros_like(channel_img)
    hls_binary[(channel_img >= thresh[0]) & (channel_img <= thresh[1])] = 1
    return hls_binary

def hsv_thresh(img, channel = 'h',thresh=(0, 255)):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    if channel == 's':
        ch = 1
    elif channel == 'v':
        ch = 2
    else:
        ch = 0
    channel_img = hsv[:,:,ch]
    hsv_binary = np.zeros_like(channel_img)
    hsv_binary[(hsv_binary > thresh[0]) & (hsv_binary <= thresh[1])] = 1
    return hsv_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize = sobel_kernel))
    sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize = sobel_kernel))
    abs_Sobel = np.sqrt(sobelx**2+sobely**2)
    scale_factor = np.max(abs_Sobel)/255 
    scaled_sobel = (abs_Sobel/scale_factor).astype(np.uint8)
    # Apply threshold
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
   # Calculate gradient magnitude
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize = sobel_kernel))
    sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize = sobel_kernel))
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    # Apply threshold
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary

def combine_rgb_thresh(img, r = (0,255), g = (0,255), b = (0,255)):
    rgb = cv2.GaussianBlur(img,(5,5),2.0)
    channel_r = rgb_thresh(rgb,'r',r)
    channel_g = rgb_thresh(rgb,'g',g)    
    channel_b = rgb_thresh(rgb,'b',b)
    rgb_binary = np.zeros_like(img)
    rgb_binary[(channel_r == 1)&(channel_g == 1)&(channel_b == 1)] = 1
    return rgb_binary

def combine_hls_thresh(img, h = (0,255), l = (0,255), s = (0,255)):
    hls = cv2.GaussianBlur(img,(5,5),2.0)
    channel_h = hls_thresh(hls,'h',h)
    channel_l = hls_thresh(hls,'l',l)    
    channel_s = hls_thresh(hls,'s',s)
    hls_binary = np.zeros_like(img)
    hls_binary[(channel_h == 1)&(channel_l == 1)&(channel_s == 1)] = 1
    return hls_binary

def combine_hsv_thresh(img, h = (0,255), s = (0,255), v = (0,255)):
    hsv = cv2.GaussianBlur(img,(5,5),2.0)
    channel_h = hsv_thresh(hsv,'h',h)
    channel_s = hsv_thresh(hsv,'s',s)    
    channel_v = hsv_thresh(hsv,'v',v)
    hsv_binary = np.zeros_like(img)
    hsv_binary[(channel_h == 1)&(channel_s == 1)&(channel_v == 1)] = 1
    return hsv_binary
    
    
    
    
