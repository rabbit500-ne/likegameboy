import cv2
import argparse
import numpy as np
import dataclasses

import blend

def create_midle_point(points):
    ret = np.array([],dtype = "int32")
    for i, point in enumerate(points):
        ret = np.append(ret, point)
        try:
            B = points[i+1]
        except IndexError:
            B = points[0]
        ret = np.append(ret, (point + B)/2)
    length = len(ret)
    ret = ret.astype(np.int32)
    return ret.reshape(int(length/2),2)

def random_point(center,cell,avr,stndr_dvtion):
    average = cell * avr
    standard_deviation = cell * stndr_dvtion
    x = center[0] + int(np.random.normal(average, standard_deviation, 1)[0])
    y = center[1] + int(np.random.normal(average, standard_deviation, 1)[0])
    return [x,y]

def angle(a,b,c):
    # ベクトルを定義
    vec_a = np.array(a) - np.array(b)
    vec_c = np.array(c) - np.array(b)

    # コサインの計算
    length_vec_a = np.linalg.norm(vec_a)
    length_vec_c = np.linalg.norm(vec_c)
    inner_product = np.inner(vec_a, vec_c)
    cos = inner_product / (length_vec_a * length_vec_c)

    # 角度（ラジアン）の計算
    rad = np.arccos(cos)

    # 弧度法から度数法（rad ➔ 度）への変換
    degree = np.rad2deg(rad)
    return degree

def round_off_corners(points, angle_th):
    servive = []
    for i, point in enumerate(points):
        try:
            pre = points[i-1]
        except IndexError:
            pre = points[-1]

        try:
            back = points[i+1]
        except IndexError:
            back = point[0]
        if angle(pre,point,back) > angle_th:
            servive.append(point)
    return np.array(servive,dtype=np.int32)

def shift(points, ic, ir, cell, margin):
    return np.array([[p[0]+ic*cell, p[1]+ir*cell] for p in points])

def main4(opt):
    """ノイズ作成"""
    image_path = opt.img
    if image_path is None:
        image_path = "../srcData/zebras-4386880_640.jpg"
    img = cv2.imread(image_path)
    height,width=img.shape[:2]
    
    noise_level=150
    noise=np.random.normal(0,20,np.shape(img))
    noise=noise.astype(np.uint8) 
    noise=cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)
    noise = cv2.blur(noise,(10,10))
    threshold = 120
    r, noise = cv2.threshold(noise, threshold, 255, cv2.THRESH_BINARY)
    # img = img - noise
    # img=noise.astype(np.uint8) 


    cv2.imshow("ccolor",noise)
    cv2.waitKey(0)

def main5(opt):
    """下地層作成"""
    """日焼け"""
    image_path = opt.img
    if image_path is None:
        image_path = "../srcData/zebras-4386880_640.jpg"
    img = cv2.imread(image_path)
    height,width,_ = np.shape(img)
    img = np.full(np.shape(img), 255, dtype=np.uint8)
    # cv2.ellipse(img, ((int(w/2), int(h/2)), (w, h), 0), (255,255,255), thickness=-1, lineType=cv2.LINE_8)
    # img = cv2.blur(img,(30,30))
    for h in range(height):
        for w in range(width):
            c = int((w+50)/width * 255)
            c = c if c <= 255 else 255
            p = c * 1.2 if c*1.2 <=255 else 255
            img[h,w]=[c,p,p]

    noise=np.random.normal(0,20,np.shape(img))
    noise=noise.astype(np.uint8) 
    # noise=cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)
    img = blend.ColorDodge(img, noise)

    
    cv2.imshow("ccolor",img)
    cv2.waitKey(0)

def main6(opt):
    BACK_PAD_COLOR=(133,187,187)#BGR
    """下地層作成"""
    image_path = opt.img
    if image_path is None:
        image_path = "../srcData/zebras-4386880_640.jpg"
    img = cv2.imread(image_path)
    height,width,_ = np.shape(img)
    img = blend.backimg(width,height)
    cv2.imshow("ccolor",img)
    cv2.waitKey(0)

def main7(opt):
    """ガラス作成"""
    img = cv2.imread("../srcData/zebras-4386880_640.jpg")
    height,width,_ = np.shape(img)
    back_img = blend.backimg(width,height)

    glass_img = np.full((height, width, 3), 255, dtype=np.uint8)
    print(f"{width}, {height}")

    # rec_points = np.array([(600,30),(630, 5), (640, 120), (620, 90)])
    # rec_points = np.array([(550,10),(630, 5), (640, 120), (540, 180)])
    # cv2.fillPoly(glass_img, [rec_points], (255, 255, 255))
    # glass_img = cv2.blur(glass_img,(100,100))
    for h in range(height):
        for w in range(width):
            c = int(((w -100)/ 2 /width) * 255)
            c = c if c <= 255 else 255
            p = int(((h -100) /height) * 255)
            p = p if p <= 255 else 255
            glass_img[h,w]=[244,2,2]


    cv2.imshow("ccolor",glass_img)
    cv2.waitKey(0)


@dataclasses.dataclass
class mask_size():
    w : int
    h : int
    d : int

    @property
    def shape(self):
        return (self.w,self.h,self.d)

class dots_layer():
    def __init__(self,path):
        self.create_canpas(path)
        self.create_dots_mask()
        self.dots_noise_mask = self.create_dots_noise_mask()
        self.dots_noise_mask2 = self.create_dots_noise_mask()
        self.dots_noise_mask3 = self.create_dots_noise_mask()
        self.create_dots_base()
        # self.create_dots_noise_color()

    def create_canpas(self,path):
        self.__margin = 10
        self.__cell = 20
        self.__cell_margin = 1
        
        self.src = cv2.imread(path)
        (src_w, src_h, src_d) = self.src.shape
        print(f"{src_w}, {src_h}, {src_d}")
        self.canpas_size = mask_size(
            w=src_w * self.__cell + self.__margin *2, h=src_h * self.__cell + self.__margin*2,d=3)

    def dot_write(self, img, ic, ir, c):
        if c == 0:
            return
        cell_margin_p = self.__cell - self.__cell_margin
        ret = np.array([(self.__cell_margin, self.__cell_margin), (cell_margin_p, self.__cell_margin), (cell_margin_p, cell_margin_p), (self.__cell_margin, cell_margin_p)])
        ret = create_midle_point(ret)
        ret = create_midle_point(ret)
        ret = create_midle_point(ret)
        ret = create_midle_point(ret)

        ret = np.array([ random_point(p, self.__cell, 0.002,0.002)for p in ret ])

        ret = shift(ret, ic, ir, self.__cell, self.__margin)
        ret = round_off_corners(ret, 95)
        cv2.fillPoly(img, [ret], (0, 0, 0))

    def create_dots_mask(self):
        self.dots_mask = np.full((self.canpas_size.w, self.canpas_size.h), 255, dtype=np.uint8)
        zero = np.array([3,3,3])
        for ir, r in enumerate(self.src):
            for ic, c in enumerate(r):
                c = 1 if np.all(c<zero) else 0
                self.dot_write(self.dots_mask, ic, ir,c )

        # self.dots_mask = blend.resize(self.dots_mask, 0.08,0.08)
        # cv2.imshow("ccolor",self.dots_mask)
        # cv2.waitKey(0)

    def create_dots_noise_mask(self):
        noise_level=150
        print(f"canpas {self.canpas_size.shape}")
        noise=np.random.normal(0,20,self.canpas_size.shape)
        print(f"noise {noise.shape}")
        noise=noise.astype(np.uint8) 
        noise=cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)
        noise = cv2.blur(noise,(10,10))
        threshold = 120
        r, dots_noise_mask = cv2.threshold(noise, threshold, 255, cv2.THRESH_BINARY)
        # print(f"ret {dots_noise_mask.shape}")
        return dots_noise_mask

    def create_dots_base(self):
        self.dots_base = np.full((self.canpas_size.w, self.canpas_size.h), 255, dtype=np.uint8)
        for ir, r in enumerate(self.src):
            for ic, c in enumerate(r):
                self.dot_write(self.dots_base, ic, ir, 1 )

    def create_dots_noise_color(self):
        img = blend.alfa_blend_mask(self.dots_mask, self.dots_base, 0.3)

        # img = cv2.addWeighted(self.dots_noise_mask | self.dots_base, 0.4, self.dots_noise_mask2 | self.dots_base, 0.6, 0)
        # img = cv2.addWeighted(self.dots_noise_mask3 | self.dots_base, 0.4, img, 0.6, 0)
        return img




class ground_layer():
    def __init__(self, shape):
        self.shape = shape # (w,h,d)
        self.create_ground()

    def create_ground(self):
        (w,h,d)=self.shape
        # self.img = np.full((w,h), 255, dtype=np.uint8)
        noise=np.clip(np.random.normal(240,10,(w,h)), 0, 255).astype(np.uint8)
        base = np.full((w,h,3), (139, 200, 190), dtype=np.uint8)
        self.img = blend.Multiply(base, cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR))
        
        noise2=np.clip(np.random.normal(100,30,(w,h)), 0, 255).astype(np.uint8)
        mask = blend.line_gradation(noise, (-1, 4, 100), (0, 200), 300)
        noise2 = blend.mask(noise2, mask)
        base2 = np.full((w,h,3), (35, 50, 58), dtype=np.uint8)
        base2 = blend.mask_color(base2, noise2)

        mask = blend.line_gradation(noise, (-1, 4, 200), (0, 200), 400)
        base3 = np.full((w,h,3), (34, 51, 57), dtype=np.uint8)
        base3 = blend.mask_color(base3, mask)


        self.img = blend.Multiply(self.img, base2)
        # self.img = blend.Multiply(self.img, base3)

        # mask = blend.alfa_blend_mask(self.img, noise2, 0.5)

        # self.img = mask
        # self.img = cv2.blur(self.img,(5,5))
        # self.ground = blend.get_gradient_2d(1,100,w,h,True)
        # self.ground2 = blend.get_gradient_3d(w, h, (0, 0, 192), (255, 255, 255), (True, False, True))

class enclosure():
    def __init__(self, img):
        self.shape = img.shape
        self.create(img)

    def create(self, src):
        PADDING = 3
        CIRVE = 10
        ENCLOSURE_COLOR = (91, 93, 94)
        p_c = PADDING + CIRVE
        (w,h,d) = self.shape
        base = np.full(self.shape, ENCLOSURE_COLOR, dtype=np.uint8)
        base = cv2.rectangle(base, (p_c, p_c), (w-p_c,w-p_c), (0, 0, 0), thickness=CIRVE)
        base = cv2.rectangle(base, (p_c, p_c), (w-p_c,w-p_c), (0, 0, 0), thickness=-1)
        
        light_base = self.create_light(base,ENCLOSURE_COLOR)
        light_circle = self.create_light_circle()
        light_base = blend.mask_color2(cv2.cvtColor(light_circle,cv2.COLOR_GRAY2BGR), light_base  )
        self.light = self.light(base, (255,255,255), light_base)
        self.img = blend.addition(base, self.light)

    def create_light(self, enclosre_img, color):
        black_img = np.full(self.shape, (0,0,0), dtype=np.uint8)
        glay_img = np.full(self.shape, color, dtype=np.uint8)
        white_img = np.full(self.shape, (255,255,255), dtype=np.uint8)
        msk_flg = enclosre_img == (0,0,0)
        non_mask_flg = enclosre_img != (0,0,0)
        light_tmp = np.full(self.shape, (91, 93, 94), dtype=np.uint8)
        light_tmp[msk_flg] = glay_img[msk_flg]
        light_tmp[non_mask_flg] = black_img[non_mask_flg]

        ret = np.full(self.shape, (0,0,0), dtype=np.uint8)

        fil = blend.shift(light_tmp, (1,-1))
        tmp = np.clip(enclosre_img + fil, a_min=0, a_max=255)
        flg = tmp > color
        ret[flg] = white_img[flg]
        return ret
        
    def create_light_circle(self):
        base = np.full(self.shape, (0,0,0), dtype=np.uint8)
        base = blend.circle_gradation(base, (190,10), 0, 100)
        return base

    def light(self,img, light_color, light_mask):
        light_base = np.full(self.shape, light_color, dtype=np.uint8)
        light_base = blend.mask_color2(light_base, light_mask)
        return light_base

class glass():
    def __init__(self, enclosure):
        self.enclosure = enclosure
        self.create()
    
    def create(self):
        self.shape = self.enclosure.shape
        self.enclosure_light = blend.shift(self.enclosure.light, (0, 3))
        self.img = self.enclosure_light
        
        glass_img = np.full(self.shape, 0, dtype=np.uint8)
        print(self.shape)
        # rec_points = np.array([(200,30),(290, 70), (290, 230), (160, 160)])
        rec_points = np.array([(150,10),(290, 70), (290, 230), (10, 30)])
        cv2.fillPoly(glass_img, [rec_points], (255, 255, 255))
        rec_points = np.array([(220,10),(230, 10), (130, 230), (120, 230)])
        cv2.fillPoly(glass_img, [rec_points], (0, 0, 0))
        rec_points = np.array([(30,10),(35, 10), (300, 150), (300, 155)])
        cv2.fillPoly(glass_img, [rec_points], (0, 0, 0))
        tmp = blend.line_gradation(glass_img,(100,1,-200), (0,300), 400)
        glass_img = blend.Multiply(glass_img, tmp)
        self.img = glass_img



class gameboy():
    def __init__(self):
        # self.dots_layer = dots_layer("../srcData/testpic.jpg")
        self.ground = ground_layer((300,300,3))
        self.en = enclosure(self.ground.img)
        self.gls = glass(self.en)
        self.blend()

    def blend(self):
        tmp = blend.addition(self.ground.img, self.gls.img * 0.2)
        self.img = blend.addition_mask(tmp,self.en.img)




def create_dots_pic(path):
    img = np.full((15,15), 255, dtype=np.uint8)
    img[1][0] = 0
    img[1][3] = 0
    img[3][4] = 0
    cv2.imwrite(path, img)

def create_circle_pic(path):
    img = np.full((200,200,3), 255, dtype=np.uint8)
    cv2.circle(img, (80,100), 40, (131,101,255),thickness=-1)
    cv2.imwrite(path, img)

def main8(opt):
    # src = cv2.imread("../srcData/sample.png")
    dots_ly = dots_layer("../srcData/testpic.jpg")
    # ground_ly = ground_layer(dots_ly)

    img = cv2.addWeighted(dots_ly.dots_noise_mask | dots_ly.dots_base, 0.05, dots_ly.dots_mask, 0.95, 0)
    img = dots_ly.create_dots_noise_color()
    dots_mask = blend.resize(np.uint8(img), 1.0,1.0)
    cv2.imshow("ccolor",dots_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main9(opt):
    gb = gameboy()
    cv2.imwrite("../srcData/noise.jpg", gb.img)
    # cv2.imshow("ccolor",ground_ly.img)
    # cv2.waitKey(0)


"""
TODO
ピクセル層
・ピクセル描写 70%
・ピクセル残滓描写 70%

下地層
・下地描写
・日焼け表現
・枠影
・ピクセル層影

ガラス層
・光反射
・傷


資料
フィルタ
https://optie.hatenablog.com/entry/2018/03/15/212107

"""

def main3(opt):
    array = [[0,0,0,0,0,0],[0,1,0,1,1,0],[0,1,0,1,1,0],[0,1,0,1,1,0],[0,1,0,1,1,0],[0,1,1,1,1,0]]
    view(array)

def main2(opt):
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (180, 180), (255, 0, 0), thickness=-1)
    correct_points = np.array([(20, 20), (180, 20), (180, 180), (20, 180)])
    print(correct_points)
    ret = create_midle_point(correct_points)
    ret2 = np.array([ random_point(p)for p in ret ])

    print(ret2)
    cv2.fillPoly(img, [ret2], (255, 255, 0))

    cv2.imshow("ccolor",img)
    cv2.waitKey(0)

def main(opt):
    image_path = opt.img
    if image_path is None:
        image_path = "../srcData/zebras-4386880_640.jpg"
    img_color = cv2.imread(image_path)
    cv2.imshow("ccolor",img_color)
    cv2.waitKey(0)


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default=None, help='pictuer path')
    
    opt = parser.parse_args()
    main9(opt)

