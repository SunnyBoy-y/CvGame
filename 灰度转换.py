# 导入cv
import cv2 as cv
# 读取模块 imread读取图片
img = cv.imread("face1.png")
# 灰度转换
gray_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# 显示灰度
cv.imshow("gray.img",gray_img)
# 保存灰度图片
cv.imwrite("save_gray.png", gray_img)
# 等待 cv.waitKey() 0表示无限等待，2表示2ms
cv.waitKey(0)
# 释放内存
cv.destroyAllWindows()
