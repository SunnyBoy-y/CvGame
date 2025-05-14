# 导入cv
import cv2 as cv

def face_detect_demo():
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detect = cv.CascadeClassifier
# 读取模块 imread读取图片
img = cv.imread("face1.png")
# 检测函数
face_detect_demo()

# cv识别键盘退出
while True:
    if ord('q') == cv.waitKey(0):
        break
# 释放内存
cv.destroyAllWindows()
