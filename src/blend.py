"""

"""

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
	shape = img.shape
	return cv2.resize(img , (int(shape[0]*recio_w), int(shape[1]*recio_h)))

def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T

def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=np.float)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

    return result

def alfa_blend(base, layer, ratio):
    print(f"base.shape{base.shape}")
    if ratio < 0 or ratio > 1:
        raise Exception()
    ret = base.copy()
    masked = np.where((layer <= [251,251,251]).all(axis=2))
    ret[masked] = ret[masked] * (1-ratio) + layer[masked] * ratio
    print(f"ret.shape{ret.shape}")
    return ret