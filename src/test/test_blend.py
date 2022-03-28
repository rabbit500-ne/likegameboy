import pytest
import sys
import cv2

sys.path.append("../")
import blend
from utils import *

DATA_DIR = "./data/test_blend/"

TEST_DISTANCE = [
    ((0,0),(1,0),1),
    ((1,1),(1,1),0),
    ((1,3),(1,2),1),
    ((1,1),(4,5),5),
]

TEST_LINE_DISTANCE = [
    ((0, 1, -1), ( 1, 2), 1),
    ((0, 1, -1), ( 1, 3), 2),
    ((1, 0, -2), ( 1, 2), -1),
    ((1, 0, -2), ( 3, 2), 1),
    ((-3, 4, 0), ( 0, 5), 4),
]

@pytest.mark.parametrize('A, B, dis', TEST_DISTANCE )
def test_distance(A,B, dis):
    d = blend.distance(A,B)
    assert d == dis

@pytest.mark.parametrize('line, point, correct', TEST_LINE_DISTANCE)
def test_line_distance(line, point, correct):
    d = blend.line_distance(line, point)
    assert d == correct

def test_line_gradation():
    """ 動作のみ確認なし """
    img = cv2.imread(pthj(DATA_DIR, "testCircle.jpg"))
    line = (4, -3, 0)
    nomal_pic = ( 1,1)
    range = 50
    img = blend.line_gradation(img, line, nomal_pic, range)
    # cv2.imshow("ccolor",img)
    # cv2.waitKey(0)

def test_circle_gradation():
    """ 動作のみ確認なし """
    img = cv2.imread(pthj(DATA_DIR, "testCircle.jpg"))
    img = blend.circle_gradation(img, (100,100), 10, 100)
    # cv2.imshow("color", img)
    # cv2.waitKey(0)