import os
from venv import create

os.environ['GLOG_minloglevel'] = '3'
import cv2 as cv
import numpy as np
import mediapipe as mp
import random
import time




class Game:
    def __init__(self):
        self.cap=cv.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.score=0

        self.mouse_region=1
        self.start_time=time.time()
        self.current_time=self.start_time
        self.duration_time=30

        self.mouse_time=self.current_time
        self.has_mouse=False
        self.mouse_duration=4

        self.feedback_time=self.mouse_time
        self.score_active=True

        self.fireworks_active=False
        self.firework_time=self.current_time
        self.firework_region=self.mouse_region
        self.gou_active=False
        self.cha_active=False
        self.gou_region=self.mouse_region
        self.cha_time=self.mouse_time



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
    def create_mouse(self,w,h,photo):
        '''
        # 随机生成老鼠
        :param w:
        :param h:
        :return:
        '''
        w=w//2
        h=h//2
        region=[
            (0,0),
            (w,0),
            (0,h),
            (w,h)
        ]
        if not self.has_mouse:
            if time.time() - self.mouse_time<0.4:
                pass
            choose=random.randint(1,4)
            while choose==self.mouse_region:
                choose=random.randint(1,4)
            else:
                self.mouse_region=choose
                self.has_mouse = True
                self.mouse_time = self.current_time
                self.mouse_duration=random.randint(3,5)
        x,y=region[self.mouse_region-1]
        self.draw_mouse(photo,x,y,w,h)
        if time.time() - self.mouse_time > self.mouse_duration:
            self.has_mouse = False


    def draw_score_time(self,photo,w,h):
        '''
        # 显示剩余时间
        :param photo:
        :param w:
        :param h:
        :return:
        '''
        cv.putText(photo, f"Remaining {int(self.duration_time-self.current_time+self.start_time)}", (10, h-30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.putText(photo, f"score {self.score}", (10+w//2, h-30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def draw_X(self,photo,w,h):
        """
        # 绘制大叉叉
        :param photo:
        :param x:
        :param y:
        :param w:
        :param h:
        :return:
        """
        w = w // 2
        h = h // 2
        region = [
            (0, 0),
            (w, 0),
            (0, h),
            (w, h)
        ]
        x, y = region[self.gou_region - 1]

        # 计算对角线的端点坐标
        pt1 = (x + 20, y + 20)
        pt2 = (x + w - 20, y + h - 20)
        pt3 = (x + w - 20, y + 20)
        pt4 = (x + 20, y + h - 20)

        # 绘制两条对角线
        cv.line(photo, pt1, pt2, (0, 0, 255), 3)  # 红色线条
        cv.line(photo, pt3, pt4, (0, 0, 255), 3)  # 红色线条

    def draw_gou(self,photo,w,h):
        '''
        # 绘制大勾勾
        :param photo:
        :param x:
        :param y:
        :param w:
        :param h:
        :return:
        '''
        w=w//2
        h=h//2
        region = [
            (0, 0),
            (w, 0),
            (0, h),
            (w, h)
        ]
        x,y=region[self.firework_region-1]
        w1=w
        h1=h
        pt1=(x+30,y+h1-40)
        pt2=(x+w1//2,y+h1-10)
        pt3=(x+w1-30,y+30)
        cv.line(photo, pt1, pt2, (0, 255, 0), 3)
        cv.line(photo, pt2, pt3, (0, 255, 0), 3)


    def draw_mouse(self, photo, x, y, w, h):
        """
        # 绘制米老鼠风格的简笔画老鼠
        :param photo: 图像帧
        :param x: 区域左上角x坐标
        :param y: 区域左上角y坐标
        :param w: 区域宽度
        :param h: 区域高度
        """
        x = x + w // 2
        y = y + h // 2
        r = min(w, h) // 6

        # 身体轮廓
        cv.circle(photo, (x, y), r * 2, (255, 0, 0), 1)
        # 头部轮廓
        cv.circle(photo, (x, y - r), r, (255, 0, 0), 1)
        # 耳朵轮廓
        cv.circle(photo, (x - r, y - r * 2), r // 2, (255, 0, 0), 1)
        cv.circle(photo, (x + r, y - r * 2), r // 2, (255, 0, 0), 1)
        # 眼睛
        cv.circle(photo, (x - r // 2, y - r + r // 3), r // 5, (255, 0, 0), 1)
        cv.circle(photo, (x + r // 2, y - r + r // 3), r // 5, (255, 0, 0), 1)
        # 尾巴
        cv.line(photo, (x - r * 2, y), (x - r * 4, y + r), (255, 0, 0), 1)


    def check_fists(self, photo, region, hands):
        """
        # 检测拳头在那个区域
        :param photo:
        :param region:
        :param hands:
        :return:
        """
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

                # 判断属于哪个大区域
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2

                height, width = photo.shape[:2]
                half_w, half_h = width // 2, height // 2

                # 判断是否在某个完整区域内
                if is_fist:
                    if center_x < half_w-40 and center_y < half_h-40:
                        return 1
                    elif center_x >= half_w-40 and center_y < half_h-40:
                        return 2
                    elif center_x < half_w-40 and center_y >= half_h-40:
                        return 3
                    elif center_x >= half_w-40 and center_y >= half_h-40:
                        return 4
                else:
                    return 0  # 在边界或无法确定
        return 0  # 未检测到手势


    def draw_fireworks(self, photo, w, h):
        '''
        # 绘画粒子效果的烟花
        :param photo:
        :param x: 区域左上角x坐标
        :param y: 区域左上角y坐标
        :param w: 区域宽度
        :param h: 区域高度
        :return:
        '''
        w = w // 2
        h = h // 2
        region = [
            (0, 0),
            (w, 0),
            (0, h),
            (w, h)
        ]
        x1,y1=region[self.firework_region-1]
        for _ in range(50):  # 粒子数量
            x = np.random.randint(x1, x1 + w)
            y = np.random.randint(y1, y1 + h)
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            cv.circle(photo, (x, y), 5, color, -1)
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
                self.current_time = time.time()
                ret,photo=self.cap.read()
                photo=cv.flip(photo,1)
                h,w=self.get_h_w(photo)
                self.draw_divide_region(photo,w,h)
                self.draw_score_time(photo,w,h)
                # self.draw_mouse(photo,0,0,w//2,h//2)
                self.create_mouse(w,h,photo)
                region = (0, 0, w, h)
                region_id=self.check_fists(photo, region,hands)
                # self.draw_gou(photo,0,0,w,h)
                # self.draw_X(photo,0,0,w,h)

                if self.cha_active and time.time()-self.cha_time<1:
                    self.draw_X(photo,w, h)
                else:
                    self.cha_active=False

                if self.gou_active and self.gou_region!=self.mouse_region and time.time()-self.firework_time<1:
                    self.draw_gou(photo, w, h)
                else:
                    self.gou_active=False

                if self.fireworks_active and time.time()-self.feedback_time<3:
                    self.draw_fireworks(photo, w, h)
                else:
                    self.fireworks_active=False
                # self.draw_firewprks(photo,0,0,w//2,h//2)
                if int(time.time() - self.start_time) > self.duration_time:
                    cv.putText(photo, f"The end!", (w//2-240, h//2), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
                    cv.imshow("photo", photo)
                    if cv.waitKey(0) == ord('q'):
                        break
                cv.imshow("photo",photo)

                # 将 score_active 改为基于 feedback_time 判断
                if region_id != 0 :
                    print(f"在区域 {region_id} 检测到拳头")

                    if region_id == self.mouse_region:
                        if time.time() - self.feedback_time > 0.6:
                            self.firework_region = region_id
                            self.feedback_time = time.time()
                            self.has_mouse = False
                            self.score += 1  # 命中加分
                            self.score_active=False
                            self.fireworks_active=True
                            self.firework_time=time.time()
                            self.gou_active=True
                            # self.cha_active=False
                            # self.cha_active=False
                    else:
                        if self.score_active:
                            self.cha_time=time.time()
                            self.gou_region = region_id
                            self.score_active=False
                            self.cha_active=True
                            # self.gou_active=False

                            self.feedback_time = time.time()
                            self.score -= 1  # 错误扣分

                if time.time()-self.feedback_time>0.6:
                    self.score_active=True



                if cv.waitKey(1) == ord('q'):
                    break

if __name__ =="__main__":
    game = Game()
    game.run()
