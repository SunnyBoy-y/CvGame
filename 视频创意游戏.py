import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import random

mp_hands = mp.solutions.hands

class Videogame:
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        self.draw_canvas = None
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.region_last_hiy_time={0:-2,1:-2,2:-2,3:-2}
        self.hit_cooldown=1.0
        # 老鼠相关
        self.mole_region = -1
        self.last_mole_region = -1  # 避免连续两次在同一个区域出现
        self.mole_start_time = 0
        self.mole_duration = 0

        # 分数统计
        self.score = 0
        self.start_time = 0
        self.game_duration = 60  # 游戏时间（秒）

        # 烟花控制
        self.fireworks_active = False
        self.fireworks_timer = 0
        self.fireworks_region = -1

        # 叉叉与对勾控制
        self.cross_region = -1
        self.check_region = -1
        self.feedback_timer = 0
        self.hit_region = -1

    def draw_mouse_simple(self, frame, x, y, w, h):
        """
        绘制简笔画风格的老鼠（仅线条）
        """
        center_x = x + w // 2
        center_y = y + h // 2
        radius = min(w, h) // 6

        # 身体轮廓
        cv.circle(frame, (center_x, center_y), radius * 2, (0, 255, 255), 1)
        # 头部轮廓
        cv.circle(frame, (center_x, center_y - radius), radius, (0, 255, 255), 1)
        # 耳朵轮廓
        cv.circle(frame, (center_x - radius, center_y - radius * 2), radius // 2, (0, 255, 255), 1)
        cv.circle(frame, (center_x + radius, center_y - radius * 2), radius // 2, (0, 255, 255), 1)
        # 眼睛
        cv.circle(frame, (center_x - radius // 2, center_y - radius + radius // 3), radius // 5, (0, 0, 0), 1)
        cv.circle(frame, (center_x + radius // 2, center_y - radius + radius // 3), radius // 5, (0, 0, 0), 1)
        # 尾巴
        cv.line(frame, (center_x - radius * 2, center_y), (center_x - radius * 4, center_y + radius), (0, 255, 255), 1)

    def draw_cross(self, frame, index, half_width, half_height):
        positions = [
            (0, 0),
            (half_width, 0),
            (0, half_height),
            (half_width, half_height)
        ]
        x, y = positions[index]
        cv.line(frame, (x + 20, y + 20), (x + half_width - 20, y + half_height - 20), (0, 0, 255), 3)
        cv.line(frame, (x + half_width - 20, y + 20), (x + 20, y + half_height - 20), (0, 0, 255), 3)

    def draw_check(self, frame, index, half_width, half_height):
        positions = [
            (0, 0),
            (half_width, 0),
            (0, half_height),
            (half_width, half_height)
        ]
        x, y = positions[index]
        pt1 = (x + 30, y + half_height - 30)
        pt2 = (x + half_width // 2, y + half_height - 10)
        pt3 = (x + half_width - 30, y + 30)
        cv.line(frame, pt1, pt2, (0, 255, 0), 3)
        cv.line(frame, pt2, pt3, (0, 255, 0), 3)

    def is_closed_fist(self, landmarks, frame_shape):
        height, width = frame_shape[0], frame_shape[1]

        def norm_to_pixel(landmark):
            return int(landmark.x * width), int(landmark.y * height)

        thumb = norm_to_pixel(landmarks[mp_hands.HandLandmark.THUMB_TIP])
        index = norm_to_pixel(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP])
        middle = norm_to_pixel(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP])
        ring = norm_to_pixel(landmarks[mp_hands.HandLandmark.RING_FINGER_TIP])
        pinky = norm_to_pixel(landmarks[mp_hands.HandLandmark.PINKY_TIP])

        palm_center_x = (thumb[0] + index[0] + middle[0] + ring[0] + pinky[0]) // 5
        palm_center_y = (thumb[1] + index[1] + middle[1] + ring[1] + pinky[1]) // 5

        threshold = 30
        return (
                abs(thumb[0] - palm_center_x) < threshold and
                abs(thumb[1] - palm_center_y) < threshold and
                abs(index[0] - palm_center_x) < threshold and
                abs(index[1] - palm_center_y) < threshold and
                abs(middle[0] - palm_center_x) < threshold and
                abs(middle[1] - palm_center_y) < threshold and
                abs(ring[0] - palm_center_x) < threshold and
                abs(ring[1] - palm_center_y) < threshold and
                abs(pinky[0] - palm_center_x) < threshold and
                abs(pinky[1] - palm_center_y) < threshold
        )

    def draw_fireworks_on_frame(self, frame, region_x, region_y, width, height):
        for _ in range(50):  # 粒子数量
            x = np.random.randint(region_x, region_x + width)
            y = np.random.randint(region_y, region_y + height)
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            cv.circle(frame, (x, y), 5, color, -1)

    def show_end_screen(self):
        # 创建一个空白图像作为弹窗
        end_window_size = (600, 400)
        end_frame = np.zeros((end_window_size[1], end_window_size[0], 3), dtype=np.uint8)

        # 绘制渐变背景
        for y in range(end_window_size[1]):
            color = (y // 5 % 200 + 55, y // 3 % 200 + 55, y // 7 % 200 + 55)
            cv.line(end_frame, (0, y), (end_window_size[0], y), color, 1)

        # 绘制边框
        cv.rectangle(end_frame, (10, 10), (end_window_size[0] - 10, end_window_size[1] - 10),
                     (255, 255, 255), 2)

        # 显示标题和分数
        title_text = "Game Over"
        score_text = f"Score: {self.score}"
        emoji_text = "\U0001F3AF"  # 🎯 靶心表情

        # 设置字体大小和位置
        font = cv.FONT_HERSHEY_SIMPLEX
        scale_title = 1.5
        scale_score = 1.2

        # 获取文本尺寸
        title_size = cv.getTextSize(title_text, font, scale_title, 2)[0]
        score_size = cv.getTextSize(score_text, font, scale_score, 2)[0]

        # 居中坐标
        title_x = (end_window_size[0] - title_size[0]) // 2
        score_x = (end_window_size[0] - score_size[0]) // 2
        emoji_x = (end_window_size[0] - len(emoji_text) * 20) // 2

        # 绘制标题
        cv.putText(end_frame, title_text, (title_x, 80), font, scale_title, (255, 255, 255), 3)

        # 绘制靶心符号
        cv.putText(end_frame, emoji_text, (emoji_x, 180), font, 2, (0, 255, 255), 2)

        # 绘制分数
        cv.putText(end_frame, score_text, (score_x, 260), font, scale_score, (255, 255, 255), 2)

        # 提示信息
        hint_text = "Press any key to exit"
        hint_size = cv.getTextSize(hint_text, font, 0.6, 1)[0]
        hint_x = (end_window_size[0] - hint_size[0]) // 2
        cv.putText(end_frame, hint_text, (hint_x, 350), font, 0.6, (200, 200, 200), 1)

        # 显示窗口并等待按键
        cv.imshow("Game Over", end_frame)
        cv.waitKey(0)
        cv.destroyWindow("Game Over")

    def run(self):
        self.start_time = time.time()
        if not self.cap.isOpened():
            print("摄像头打开失败")
            return

        with self.hands as hands:
            while True:
                ret, frame = self.cap.read()
                frame = cv.flip(frame, 1)
                if not ret:
                    print("摄像头读取失败")
                    break

                height, width = frame.shape[:2]
                half_width, half_height = width // 2, height // 2
                regions = [
                    (0, 0, half_width, half_height),
                    (half_width, 0, half_width, half_height),
                    (0, half_height, half_width, half_height),
                    (half_width, half_height, half_width, half_height)
                ]

                current_time = time.time()
                elapsed_time = current_time - self.start_time
                if elapsed_time >= self.game_duration:
                    print(f"游戏结束，得分为：{self.score}")
                    self.show_end_screen()
                    break

                # 控制老鼠刷新逻辑（不会连续出现在同一位置）
                if self.mole_region == -1 or (current_time - self.mole_start_time) > self.mole_duration:
                    available_regions = [i for i in range(4) if i != self.last_mole_region]
                    if available_regions:
                        self.mole_region = random.choice(available_regions)
                        self.last_mole_region = self.mole_region
                        self.mole_start_time = current_time
                        self.mole_duration = random.uniform(3, 5)

                # 绘制区域边框和编号
                for i, (x, y, w, h) in enumerate(regions):
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.putText(frame, f"Region {i+1}", (x + 10, y + 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 绘制老鼠
                if self.mole_region != -1:
                    x, y, w, h = regions[self.mole_region]
                    self.draw_mouse_simple(frame, x, y, w, h)

                # 绘制烟花
                if self.fireworks_active:
                    x, y, w, h = regions[self.fireworks_region]
                    self.draw_fireworks_on_frame(frame, x, y, w, h)
                    if time.time() - self.fireworks_timer > 4:
                        self.fireworks_active = False

                # 绘制对勾（✔️）或叉叉（❌）
                if self.check_region != -1:
                    self.draw_check(frame, self.check_region, half_width, half_height)
                    if time.time() - self.feedback_timer > 0.5:
                        self.check_region = -1
                        self.fireworks_active = True
                        self.fireworks_region = self.hit_region
                        self.fireworks_timer = time.time()

                elif self.cross_region != -1:
                    self.draw_cross(frame, self.cross_region, half_width, half_height)
                    if time.time() - self.feedback_timer > 0.5:
                        self.cross_region = -1

                # MediaPipe 手势识别
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        if self.is_closed_fist(hand_landmarks.landmark, frame.shape):
                            x_coords = [int(lm.x * width) for lm in hand_landmarks.landmark]
                            y_coords = [int(lm.y * height) for lm in hand_landmarks.landmark]
                            center_x, center_y = sum(x_coords) // len(x_coords), sum(y_coords) // len(y_coords)

                            region_index = -1
                            for i, (x, y, w, h) in enumerate(regions):
                                if x <= center_x < x + w and y <= center_y < y + h:
                                    region_index = i
                                    break
                            current_time = time.time()
                            if region_index != -1:
                                if current_time - self.region_last_hiy_time[region_index] > self.hit_cooldown:
                                    if self.mole_region == region_index:
                                        # 击中老鼠：✔️ → 烟花 → +1 分
                                        self.hit_region = region_index
                                        self.check_region = region_index
                                        self.score += 1
                                        self.mole_region = -1
                                        self.feedback_timer = time.time()
                                    else:
                                        # 击中空区域：❌ → -1 分
                                        self.cross_region = region_index
                                        self.score -= 1
                                        self.feedback_timer = time.time()
                                    self.region_last_hiy_time[region_index] = current_time

                # 显示得分在屏幕正中间上方
                score_text = f"Score: {self.score}"
                text_size = cv.getTextSize(score_text, cv.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                text_x = (width - text_size[0]) // 2
                cv.putText(frame, score_text, (text_x, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

                cv.imshow("Video", frame)
                if cv.waitKey(1) == ord('q'):
                    break

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    game = Videogame()
    game.run()
