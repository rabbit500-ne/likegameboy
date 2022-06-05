"""

"""
import math
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

def alfa_blend_mask(base, layer, ratio):
    print(f"base.shape{base.shape}")
    if ratio < 0 or ratio > 1:
        raise Exception()
    ret = base.copy()
    # masked = np.where((layer <= [251,251,251]).all(axis=2))
    masked = np.where(layer <= 251,True, False)
    ret[masked] = ret[masked] * (1-ratio) + layer[masked] * ratio
    print(f"_ret.shape{ret.shape}")
    return ret

def alfa_blend_color(base, layer, ratio):
    if base.shape[2] != 3 or layer.shape[2] != 3:
        raise Exception
    ret = base.copy()
    masked = np.where((layer <= [255,255,255]).all(axis=2))
    ret[masked] = ret[masked] * (1-ratio) + layer[masked] * ratio
    print(f"c_ret.shape{ret.shape}")
    return ret

def alfa_convert(base, alfa_img):
    if base.shape[2] != 3 or len(alfa_img.shape) != 2:
        raise Exception
    ch_b, ch_g, ch_r = cv2.split(base[:,:,:3])
    return cv2.merge((ch_b, ch_g, ch_r, alfa_img))

def distance(A, B):
    return np.linalg.norm(np.array(A)-np.array(B), ord=2)

def circle_nomal_vec_point(cx, cy, r, x1, y1, x2, y2):
    '''円と直線の交点'''
    xd = x2 - x1; yd = y2 - y1
    X = x1 - cx; Y = y1 - cy
    a = xd**2 + yd**2
    b = xd * X + yd * Y
    c = X**2 + Y**2 - r**2
    # D = 0の時は1本で、D < 0の時は存在しない
    D = b**2 - a*c
    if D < 0 or a == 0:
        raise ValueError

    s1 = (-b + math.sqrt(D)) / a
    s2 = (-b - math.sqrt(D)) / a
    p1 = (x1 + xd*s1, y1 + yd*s1)
    p2 = (x1 + xd*s2, y1 + yd*s2)
    if (cx - x1) * (cx - x2) > 0:  # TODO ???
        return p1, p2
    else :
        return p2, p1


def circle_gradation(img, center, start_renge, end_range):
    print(f"start_range: {start_renge}")
    ret = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)
    def func(x,y):
        try:
            circle_p,_  = circle_nomal_vec_point(*center, end_range, x,y, *center)
        except ValueError:
            if (x,y) == center:
                return 255
            return 0
        dis_o_p = distance((x,y), circle_p)
        dis_p_c = distance((x,y), center)
        if dis_o_p < 0 or dis_p_c < 0:
            breakpoint() 
        dis_o_c = dis_o_p + dis_p_c
        # print(f"{[dis_p_c,start_renge]}")
        if dis_p_c <= start_renge:
            color = 255
        # elif dis_o_p > end_range:
            # color = 255
        elif dis_p_c > end_range:
            color = 0
        else :
            color = 255 - int(dis_p_c / dis_o_c * 255)
        return color
    for ri, r in enumerate(ret):
        for ci, c in enumerate(r):
            ret[ri,ci] = func(ri,ci)
    
    return ret
    
def line_distance(line, point):
    """
    line : ax + by + c = 0 が成り立つ(a,b,c)
    point : (x, y)
    """
    a,b,c = line
    x,y = point
    numer = a*x + b*y + c #分子
    denom = math.sqrt(pow(a,2)+pow(b,2)) #分母
    return numer/denom #計算結果
 
def line_gradation(img, line, nomal_pic, range):
    """
    line : ax + by + c = 0 が成り立つa,b,c
    nomal_pic : 方向性
    """
 
    (xn, yn) = nomal_pic
    if len(img.shape) == 3:
        ret = np.full(img.shape, (255, 255, 255), dtype=np.uint8)
    else :
        ret = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)

    def func(x, y):
        return line_distance(line, (x, y))
 
    nomal_flg = 1 if (func(xn,yn) >= 0) else -1
    print(nomal_flg)
    for ri, r in enumerate(ret):
        for ci, c in enumerate(r):
            distance = func(ri,ci) * nomal_flg / range * 255
            if distance > 255:
                distance = 255
            if distance <= 0:
                distance = 0
            ret[ri,ci] = distance
    return ret

def mask(src, mask):
    ret = np.full((src.shape[0], src.shape[1]), 255, dtype=np.uint8)
    for ri, r in enumerate(ret):
        for ci, c in enumerate(r):
            ret[ri,ci] = 255 - ((255 - src[ri,ci]) * (255 - mask[ri,ci]) / 255)
    return ret

def mask_color(src, mask):
    ret = np.full(src.shape, (255,255,255), dtype=np.float64)
    white = 1
    src  = src / 255
    mask = mask /255
    for ri, r in enumerate(ret):
        for ci, c in enumerate(r):
            ret[ri, ci] = white - ((white - src[ri,ci]) * (white - mask[ri,ci]))
    return ret * 255

def mask_color2(src, mask):
    flg = src >= mask
    flg2 = src < mask
    tmp = np.copy(src)
    src[flg] = mask[flg]
    src[flg2] = tmp[flg2]
    return src

def blend_color(base, upper, mask):
    ret = np.full(base.shape, (255,255,255), dtype=np.float64)
    white = np.array([255.,255.,255.])
    for ri, r in enumerate(ret):
        for ci, c in enumerate(r):
            ret[ri, ci] = white - ((white - upper[ri,ci]) * (255- mask[ri,ci]) / 255)

def Multiply(bg_img, fg_img):
    bg_img = bg_img / 255
    fg_img = fg_img / 255
    result = bg_img * fg_img
    return result * 255

def shift(img, nomal):
    (x,y) = nomal
    (w,h,d) = img.shape
    ret = np.full(img.shape, (0,0,0), dtype=np.uint8)
    if x < 0 and y >= 0:
        ret[0:w+x, y:h] = img[-x:w, 0:h-y]
    elif x >= 0 and y>= 0:
        ret[x:w, y:h] = img[0:w-x, 0:h-y]
    elif x < 0 and y < 0:
        ret[0:w+x, 0:h+y] = img[-x:w, -y:h]
    elif x >= 0 and y < 0:
        ret[x:w, 0:h+y] = img[0:w-x, -y:h]
    return ret

def Screen(bg_img, fg_img):
    result = np.zeros(bg_img.shape)
    result = 1 - ((1 - bg_img) * (1 - fg_img))
    return result

def addition(a_img, b_img):
    result = a_img / 255 + b_img / 255
    return (result.clip(0,1) * 255).astype(np.uint8)

def addition_mask(a_img, msk_img):
    msk_flg = msk_img != (0,0,0)
    a_img[msk_flg] = msk_img[msk_flg]
    return a_img

    