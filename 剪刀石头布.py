import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

def detect_gesture(cnt, img):
    # 创建掩膜
    hull = cv2.convexHull(cnt)
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    fingers = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            # 计算三角形边长
            a = np.linalg.norm(np.array(start) - np.array(end))
            b = np.linalg.norm(np.array(start) - np.array(far))
            c = np.linalg.norm(np.array(end) - np.array(far))

            angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))  # 角度公式

            if angle < 90 and d > 10000:
                fingers += 1

    # 增加一个手指是因为大拇指可能没被识别到
    total_fingers = fingers + 1

    if total_fingers == 0 or total_fingers == 5:
        return "stone"
    elif total_fingers == 2:
        return "scissors"
    elif total_fingers >= 3 and total_fingers <= 4:
        return "paper"
    else:
        return "unknown"

def game_result(player, computer):
    if player == computer:
        return "平局"
    elif (
        (player == "stone" and computer == "scissors") or
        (player == "scissors" and computer == "paper") or
        (player == "paper" and computer == "stone")
    ):
        return "你赢了！"
    else:
        return "你输了！"

# 加载中文字体
font_path = "SimHei.ttf"  # 确保字体文件存在
font_size = 30
font = ImageFont.truetype(font_path, font_size)

# 主程序
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[100:400, 100:400]  # 手部区域
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 1000:
            gesture = detect_gesture(cnt, roi)
            print("识别手势：", gesture)

            # 游戏部分
            choices = ["石头", "剪刀", "布"]
            computer = random.choice(choices)
            result = game_result(gesture, computer)

            # 将OpenCV图像转换为PIL图像以便绘制中文
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            # 绘制中文文本
            draw.text((50, 50), f"你: {gesture}", font=font, fill=(255, 0, 0))
            draw.text((50, 100), f"电脑: {computer}", font=font, fill=(0, 255, 0))
            draw.text((50, 150), f"结果: {result}", font=font, fill=(0, 0, 255))

            # 将PIL图像转换回OpenCV图像
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # 绘制轮廓
            cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)

    # 显示画面
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)
    cv2.imshow("剪刀石头布", frame)

    if cv2.waitKey(1) == 27:  # 按 ESC 键退出
        break

cap.release()
cv2.destroyAllWindows()
