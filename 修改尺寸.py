# 导入cv
import cv2 as cv
# 读取模块 imread读取图片
img = cv.imread("face1.png")
# 修改尺寸
resize_img = cv.resize(img, (320, 240))
cv.imshow("img", img)
cv.imshow("resize_img", resize_img)
cv.imwrite("resize_img.jpg", resize_img)
print(resize_img.shape)
print(img.shape)
# cv识别键盘退出
while True:
    if ord('q') == cv.waitKey(0):
        break
# 释放内存
cv.destroyAllWindows()
