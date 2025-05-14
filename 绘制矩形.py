# 导入cv
import cv2 as cv
# 读取模块 imread读取图片
img = cv.imread("face1.png")
# 坐标
x, y, w, h = 100, 100, 100, 100
# 绘制矩形
cv.rectangle(img, (x, y, x+w, y+h), color=(0, 0, 255), thickness=1)
# 绘制圆形
cv.circle(img, center=(x+w, y+h), radius=100, color=(255, 0, 0))
# 显示
cv.imshow("draw_img", img)
# cv识别键盘退出
while True:
    if ord('Q') == cv.waitKey(0):
        break
# 释放内存
cv.destroyAllWindows()
