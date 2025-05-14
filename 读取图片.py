# 导入cv
import cv2 as cv
# 读取模块 imread读取图片
img = cv.imread("face1.png")
# 显示模块
cv.imshow("read_img", img)
# 等待 cv.waitKey() 0表示无限等待，2表示2秒
cv.waitKey(0)
# 释放内存
cv.destroyAllWindows()
