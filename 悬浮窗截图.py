import cv2
import pygame
import random
import numpy as np
import mediapipe as mp

# 初始化 Pygame
pygame.init()

# 设置窗口大小和标题
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("手势识别打地鼠 - 手势1~5识别")

# 加载字体
font = pygame.font.SysFont("simhei", 36)  # 支持中文

# 颜色定义
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
GREEN = (0, 200, 0)
RED = (255, 0, 0)

# Mediapipe 手势识别初始化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# 游戏变量
score = 0
time_left = 60
mole_visible = False
mole_pos = None
hole_positions = [(x * 150 + 50, HEIGHT // 2) for x in range(5)]  # 5 个地鼠水平排列
clock = pygame.time.Clock()
game_over = False
next_mole_time = pygame.time.get_ticks() + random.randint(1000, 2000)

# 锤子初始位置
hammer_center = (WIDTH // 2, HEIGHT // 2)

# 手指编号映射：拇指 ~ 小指
FINGER_TIPS = [4, 8, 12, 16, 20]  # 拇指、食指、中指、无名指、小拇指
FINGER_DIPS = [3, 6, 10, 14, 18]

# 判断手势数字（1~5）
def recognize_gesture(landmarks):
    count = 0
    for tip, dip in zip(FINGER_TIPS, FINGER_DIPS):
        if landmarks[tip].y < landmarks[dip].y:  # 手指伸直
            count += 1
    return count if count > 0 else None  # 返回 1~5 或 None（不识别）

# 绘制洞穴
def draw_holes():
    for i, pos in enumerate(hole_positions):
        label = font.render(str(i+1), True, WHITE)
        rect = label.get_rect(center=(pos[0], pos[1]-30))
        pygame.draw.circle(screen, GRAY, pos, 40)
        screen.blit(label, rect)

# 显示分数
def show_score():
    score_text = font.render(f"得分: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

# 倒计时
def show_timer():
    timer_text = font.render(f"剩余时间: {int(time_left)}", True, WHITE)
    screen.blit(timer_text, (WIDTH - 250, 10))

# ================= 新增：显示摄像头画面 =================
def show_camera_frame(frame):
    display_frame = cv2.resize(frame, (640, 480))
    cv2.imshow("摄像头画面", display_frame)

# 主循环
running = True
while running:
    screen.fill((30, 30, 30))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ret, frame = cap.read()
    if not ret:
        break

    # ========== 新增：显示摄像头画面 ==========
    show_camera_frame(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_hammer = hammer_center
    gesture_number = None  # 默认没有识别出手势

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            gesture_number = recognize_gesture(hand_landmarks.landmark)
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            x = int(wrist.x * WIDTH)
            y = int(wrist.y * HEIGHT)
            current_hammer = (x, y)

    # 更新地鼠状态
    if not game_over:
        current_time = pygame.time.get_ticks()
        if current_time >= next_mole_time and not mole_visible:
            mole_visible = True
            mole_index = random.randint(0, 4)  # 随机选择一个地鼠编号
            mole_pos = hole_positions[mole_index]
            next_mole_time = current_time + random.randint(1000, 2000)

        time_left -= clock.tick(60) / 1000
        if time_left <= 0:
            game_over = True

    # 绘图
    draw_holes()
    show_score()
    show_timer()

    if mole_visible:
        mole_rect = pygame.Rect(mole_pos[0] - 30, mole_pos[1] - 30, 60, 60)
        mole_text = font.render("🐹", True, WHITE)
        text_rect = mole_text.get_rect(center=mole_rect.center)
        screen.blit(mole_text, text_rect)

        # 如果识别到手势数字，并且与当前地鼠匹配，则加分
        if gesture_number is not None and gesture_number <= 5:
            if mole_pos == hole_positions[gesture_number - 1]:  # 地鼠索引从 0 开始
                score += 1
                mole_visible = False

    # 绘制锤子（拳头）
    pygame.draw.circle(screen, RED, current_hammer, 20)

    if game_over:
        over_text = font.render("游戏结束！你的得分是：" + str(score), True, WHITE)
        screen.blit(over_text, (WIDTH // 2 - over_text.get_width() // 2, HEIGHT // 2))

    pygame.display.flip()

# ========== 新增：退出时关闭 OpenCV 窗口 ==========
cap.release()
cv2.destroyAllWindows()
hands.close()
pygame.quit()
