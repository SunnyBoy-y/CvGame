# 导入cv
import cv2 as cv

# 读取图片
img = cv.imread("face1.png")
gray_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 初始化窗口状态


# 显示图像
cv.imshow("img", img)
img_window_open = True

cv.imshow("gray.img", gray_img)
gray_window_open = True

# 保存灰度图
cv.imwrite("save_gray.png", gray_img)

# 主循环
while True:
    key = cv.waitKey(0)

    if key == ord('Q'):
        break

    elif key == ord('A'):
        if img_window_open:
            cv.destroyWindow("img")
            img_window_open = False
        # 可选：重新显示灰度图（确保窗口存在）
        if not gray_window_open:
            cv.imshow("gray.img", gray_img)
            gray_window_open = True

    elif key == ord('D'):
        if gray_window_open:
            cv.destroyWindow("gray.img")
            gray_window_open = False
        # 可选：重新显示原图
        if not img_window_open:
            cv.imshow("img", img)
            img_window_open = True

# 释放资源
cv.destroyAllWindows()
