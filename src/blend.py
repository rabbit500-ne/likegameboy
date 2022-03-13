import numpy as np
import cv2

def ColorDodge(bg_img, fg_img):
    result = np.zeros(bg_img.shape)
    
    fg_reverse = 1 - fg_img
    non_zero = fg_reverse!=0
    
    result[non_zero] = bg_img[non_zero]/fg_reverse[non_zero]
    result[~non_zero] = 1
    
    return result

def backimg(width, height):
    BACK_PAD_COLOR=(133,187,187)#BGR
    return np.full((height, width, 3), BACK_PAD_COLOR, dtype=np.uint8)

def resize(img, recio_w, recio_h):
	(w,h,d) = img.shape
	return cv2.resize(img , (int(w*recio_w), int(h*recio_h)))