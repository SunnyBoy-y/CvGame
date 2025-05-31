import cv2 as cv
import numpy as np
from cv2.version import opencv_version

from 视频入门 import draw_canvas


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

        :return:
        """

        x_center = x+w//2
        y_center = y+h//2

        cv.line(photo,)

    def run(self):
        if not self.open_video_check():
            return
        """
        # ret 表示cap对象的第一个返回值，True表示摄像头读取成功，False失败
        # photo 表示numpy数组的图像帧
        # cv.flip 表示反转图像
        """
        ret,photo=self.cap.read()
        photo=cv.flip(photo,1)
        w,h=self.get_h_w(photo)
        self.draw_divide_region(photo,w,h)







if __name__ =="__main__":
    game = Game()
    game.run()
