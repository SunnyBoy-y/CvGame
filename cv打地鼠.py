import os
os.environ['GLOG_minloglevel'] = '3'
import cv2 as cv
import numpy as np
import mediapipe as mp




class Game:
    def __init__(self):
        self.cap=cv.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

    def open_video_check(self):
        # 检测摄像头是否打开
        if not self.cap.isOpened():
            print("摄像头打开失败")
            return False
        else:
            return True

    def get_h_w(self,photo):
        """
        # 获取高度和宽度
        :param photo:
        :return: w 宽度,h 高度
        """
        return photo.shape[:2]

    def check_fists(self, photo, region,hands):
        x, y, w, h = region
        position = photo[y:y + h, x:x + w]
        photo_rgb = cv.cvtColor(position, cv.COLOR_BGR2RGB)
        res = hands.process(photo_rgb)
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                # 将局部坐标转换为全局坐标
                landmarks = np.array([[lm.x * w + x, lm.y * h + y] for lm in hand_landmarks.landmark])
                tip_ids = [8, 12, 16, 20]  # 手指指尖索引
                wrist = landmarks[0]
                sum_dist = 0
                for tip_id in tip_ids:
                    dist = np.linalg.norm(landmarks[tip_id] - wrist)
                    sum_dist += dist
                avg_dist = sum_dist / len(tip_ids)

                # 握拳判断
                is_fist = avg_dist < 80

                # 绘制手势关键点
                self.mp_draw.draw_landmarks(photo, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # 绘制包围手势的矩形框
                x_min, y_min = np.min(landmarks, axis=0).astype(int)
                x_max, y_max = np.max(landmarks, axis=0).astype(int)
                cv.rectangle(photo, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (255, 0, 0), 2)

                # 显示是否握拳
                label = "Fist" if is_fist else "Open"
                cv.putText(photo, label, (x_min, y_min - 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                return is_fist  # 返回当前是否握拳
        return False  # 未检测到手势

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
        # 运行模块
        ：ret 表示cap对象的第一个返回值，True表示摄像头读取成功，False失败
        ：photo 表示numpy数组的图像帧
        ：cv.flip 表示反转图像
        """
        with self.mp_hands.Hands(static_image_mode=False,
                                 max_num_hands=2,
                                 min_detection_confidence=0.6) as hands:
            while True:
                ret,photo=self.cap.read()
                photo=cv.flip(photo,1)
                h,w=self.get_h_w(photo)
                self.draw_divide_region(photo,w,h)
                region = (0, 0, w, h)
                self.check_fists(photo, region,hands)
                cv.imshow("photo",photo)
                if cv.waitKey(10) == ord('q'):
                    break






if __name__ =="__main__":
    game = Game()
    game.run()
