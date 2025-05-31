import cv2 as cv
import numpy as np
from PIL import Image,ImageDraw,ImageFont
img =np.zeros((512,512,3),np.uint8)
def event_mouse(event,x,y,flags,param):

    if event ==cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img,(x,y),50,(255,0,0),-1)


#获取视频流,返回值为一个VideoCapture对象
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

#创建掩膜
draw_canvas =None
while True:
    cv.imshow("happy",img)
    # 读取一帧图像,ret表示：读取成功返回True，读取失败返回False，frame：读取的图像数据BGR
    ret,frame=cap.read()
    # 1:左右 0：上下 -1：上下左右
    cvt_frame=cv.flip(frame,1)
    if not ret:
        print("无法获取画面")
        break

    if draw_canvas is None:
        draw_canvas = np.zeros_like(cvt_frame)

    combied = cv.addWeighted(cvt_frame,1.0,draw_canvas,1.0,0)

    #  将BGR图像转换为灰度图
    gray = cv.cvtColor(frame,cv.COLOR_BGRA2GRAY)
    cv.imshow("帧",combied)
    cv.setMouseCallback("帧",event_mouse,draw_canvas)
    if cv.waitKey(1)==ord('q'):
        break
cap.release()
cv.destroyAllWindows()