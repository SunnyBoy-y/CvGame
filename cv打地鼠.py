import cv2 as cv
import numpy as np




class Game:
    def __init__(self):
        self.cap=cv.VideoCapture(0)

    def open_video_check(self):
        # 检测摄像头是否打开
        if not self.cap.isOpened():
            print("摄像头打开失败")
            return False
        else:
            return True

    def get_h_w(self,photo):
        """
        :param photo:
        :return: w 宽度,h 高度
        """
        return photo.shape[:2]


    def draw_divide_region(self,photo,w,h):
        """
        # 绘画四个分区
        :return:无
        """
        cv.putText(photo,f"Region 1",(10,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv.putText(photo,f"Region 2",(10+w//2,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv.putText(photo,f"Region 3",(10,30+h//2),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv.putText(photo,f"Region 4",(10+w//2,30+h//2),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv.line(photo,(0,h//2),(w,h//2),(0,255,0),2)
        cv.line(photo,(w//2,0),(w//2,h),(0,255,0),2)
        cv.rectangle(photo,(0,0),(w-1,h-1),(0,255,0),3)
    def run(self):
        if not self.open_video_check():
            return
        """
        # ret 表示cap对象的第一个返回值，True表示摄像头读取成功，False失败
        # photo 表示numpy数组的图像帧
        # cv.flip 表示反转图像
        """
        while True:
            ret,photo=self.cap.read()
            photo=cv.flip(photo,1)
            h,w=self.get_h_w(photo)
            self.draw_divide_region(photo,w,h)
            cv.imshow("photo",photo)
            if cv.waitKey(1) == ord('q'):
                break






if __name__ =="__main__":
    game = Game()
    game.run()
