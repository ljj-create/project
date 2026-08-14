"""基于 YOLOv3-tiny + MediaPipe 的实时目标检测 + 动作识别 + 光粒子特效"""

import cv2
import mediapipe as mp
import numpy as np
import os
import math
import time
import random
from PIL import Image, ImageDraw, ImageFont

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

# YOLO 参数
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_SIZE = 416

# 字体
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
FONT_24 = ImageFont.truetype(FONT_PATH, 24)
FONT_20 = ImageFont.truetype(FONT_PATH, 20)
FONT_18 = ImageFont.truetype(FONT_PATH, 18)
FONT_16 = ImageFont.truetype(FONT_PATH, 16)

# COCO 80 类中文名称
COCO_CN = {
    "person": "人", "bicycle": "自行车", "car": "汽车", "motorbike": "摩托车",
    "aeroplane": "飞机", "bus": "公交车", "train": "火车", "truck": "卡车",
    "boat": "船", "traffic light": "交通灯", "fire hydrant": "消防栓",
    "stop sign": "停车标志", "parking meter": "停车计时器", "bench": "长椅",
    "bird": "鸟", "cat": "猫", "dog": "狗", "horse": "马", "sheep": "羊",
    "cow": "牛", "elephant": "大象", "bear": "熊", "zebra": "斑马",
    "giraffe": "长颈鹿", "backpack": "背包", "umbrella": "雨伞",
    "handbag": "手提包", "tie": "领带", "suitcase": "手提箱",
    "frisbee": "飞盘", "skis": "滑雪板", "snowboard": "滑雪板",
    "sports ball": "球", "kite": "风筝", "baseball bat": "棒球棒",
    "baseball glove": "棒球手套", "skateboard": "滑板", "surfboard": "冲浪板",
    "tennis racket": "网球拍", "bottle": "瓶子", "wine glass": "酒杯",
    "cup": "杯子", "fork": "叉子", "knife": "刀", "spoon": "勺子",
    "bowl": "碗", "banana": "香蕉", "apple": "苹果", "sandwich": "三明治",
    "orange": "橙子", "broccoli": "西兰花", "carrot": "胡萝卜",
    "hot dog": "热狗", "pizza": "披萨", "donut": "甜甜圈", "cake": "蛋糕",
    "chair": "椅子", "sofa": "沙发", "pottedplant": "盆栽", "bed": "床",
    "diningtable": "餐桌", "toilet": "马桶", "tvmonitor": "电视",
    "laptop": "笔记本", "mouse": "鼠标", "remote": "遥控器",
    "keyboard": "键盘", "cell phone": "手机", "microwave": "微波炉",
    "oven": "烤箱", "toaster": "烤面包机", "sink": "水槽",
    "refrigerator": "冰箱", "book": "书", "clock": "时钟",
    "vase": "花瓶", "scissors": "剪刀", "teddy bear": "泰迪熊",
    "hair drier": "吹风机", "toothbrush": "牙刷",
}

# MediaPipe 初始化
mp_hands = mp.solutions.hands  # type: ignore[attr-defined]
mp_pose = mp.solutions.pose  # type: ignore[attr-defined]
mp_drawing = mp.solutions.drawing_utils  # type: ignore[attr-defined]
mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore[attr-defined]

# 颜色主题 (BGR)
C_BG = (40, 40, 40)
C_ACCENT = (255, 200, 0)
C_GESTURE = (200, 255, 0)
C_POSE = (0, 200, 255)
C_OBJ = (255, 180, 80)
C_TEXT = (240, 240, 240)
C_DIM = (120, 120, 120)


# ==================== 光粒子系统 ====================

class Particle:
    """单个光粒子"""
    __slots__ = ["x", "y", "vx", "vy", "life", "max_life", "size", "hue", "type"]

    def __init__(self, x, y, hue, ptype="trail"):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.5, 3.0) if ptype == "burst" else random.uniform(0.2, 1.5)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - (0.5 if ptype == "trail" else 0)
        self.max_life = random.randint(15, 35) if ptype == "burst" else random.randint(20, 45)
        self.life = self.max_life
        self.size = random.uniform(3, 8) if ptype == "burst" else random.uniform(2, 5)
        self.hue = hue
        self.type = ptype


class ParticleSystem:
    """粒子系统管理器"""

    def __init__(self, max_particles=800):
        self.particles: list[Particle] = []
        self.max_particles = max_particles
        self.hue_offset = 0

    def spawn_trail(self, x, y, count=3):
        """在指尖位置生成拖尾粒子"""
        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                self.particles.pop(0)
            hue = (self.hue_offset + random.uniform(-15, 15)) % 180
            self.particles.append(Particle(x, y, hue, "trail"))

    def spawn_burst(self, x, y, count=20):
        """在手势切换时生成爆发粒子"""
        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                self.particles.pop(0)
            hue = random.uniform(0, 180)
            self.particles.append(Particle(x, y, hue, "burst"))

    def update(self):
        """更新所有粒子"""
        alive = []
        for p in self.particles:
            p.life -= 1
            if p.life <= 0:
                continue
            p.x += p.vx
            p.y += p.vy
            p.vy += 0.02  # 微弱重力
            p.vx *= 0.98   # 阻尼
            p.vy *= 0.98
            # 拖尾粒子有微弱漂浮感
            if p.type == "trail":
                p.vx += random.uniform(-0.15, 0.15)
                p.vy -= 0.01
            alive.append(p)
        self.particles = alive

    def draw(self, frame):
        """绘制所有粒子到帧上"""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        for p in self.particles:
            if p.x < 0 or p.x >= w or p.y < 0 or p.y >= h:
                continue

            ratio = p.life / p.max_life
            alpha = ratio ** 0.5
            size = p.size * ratio

            # HSV -> BGR 彩虹色
            hsv = np.array([[[p.hue, 220, 255]]], dtype=np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])

            cx, cy = int(p.x), int(p.y)

            # 外层光晕（大圆，低不透明度）
            glow_r = int(size * 3)
            if glow_r > 0:
                color_dim = (int(r * alpha * 0.3), int(g * alpha * 0.3), int(b * alpha * 0.3))
                cv2.circle(overlay, (cx, cy), glow_r, color_dim, -1)

            # 中层光晕
            mid_r = int(size * 1.8)
            if mid_r > 0:
                color_mid = (int(r * alpha * 0.6), int(g * alpha * 0.6), int(b * alpha * 0.6))
                cv2.circle(overlay, (cx, cy), mid_r, color_mid, -1)

            # 核心亮点
            core_r = max(1, int(size * 0.8))
            color_core = (int(r * alpha), int(g * alpha), int(b * alpha))
            cv2.circle(overlay, (cx, cy), core_r, color_core, -1)

            # 最亮中心点
            color_white = (
                min(255, int(r * alpha + 80)),
                min(255, int(g * alpha + 80)),
                min(255, int(b * alpha + 80)),
            )
            cv2.circle(overlay, (cx, cy), max(1, core_r // 2), color_white, -1)

        # 混合叠加
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        return frame

    def set_hue(self, hue):
        self.hue_offset = hue


# ==================== PIL 中文文字渲染 ====================

def put_cn_text(frame, text, pos, font, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    color_rgb = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def get_text_size(text, font):
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


# ==================== YOLO 目标检测 ====================

def load_yolo():
    cfg = os.path.join(MODEL_DIR, "yolov3-tiny.cfg")
    weights = os.path.join(MODEL_DIR, "yolov3-tiny.weights")
    names = os.path.join(MODEL_DIR, "coco.names")
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    with open(names, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in np.array(net.getUnconnectedOutLayers()).flatten()]
    return net, class_names, output_layers


def detect_objects(net, output_layers, frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence < CONFIDENCE_THRESHOLD:
                continue
            cx, cy, bw, bh = detection[:4]
            x = int((cx - bw / 2) * w)
            y = int((cy - bh / 2) * h)
            boxes.append([x, y, int(bw * w), int(bh * h)])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    results = []
    if len(indices) > 0:
        for i in np.array(indices).flatten():
            results.append({"box": boxes[i], "confidence": confidences[i], "class_id": class_ids[i]})
    return results


# ==================== 手势识别 ====================

def finger_status(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    fingers = []
    thumb_tip = hand_landmarks.landmark[tips[0]]
    thumb_ip = hand_landmarks.landmark[pips[0]]
    fingers.append(thumb_tip.x < thumb_ip.x)
    for tip, pip in zip(tips[1:], pips[1:]):
        fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)
    return fingers


def recognize_gesture(hand_landmarks):
    fs = finger_status(hand_landmarks)
    thumb, index, middle, ring, pinky = fs

    if thumb and not index and not middle and not ring and not pinky:
        return "竖大拇指"
    if index and middle and not ring and not pinky:
        return "比耶"
    if all(fs):
        return "张开手掌"
    if not any(fs):
        return "握拳"
    if index and not middle and not ring and not pinky:
        return "指向"

    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    dist = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    if dist < 0.05 and not middle and not ring and not pinky:
        return "OK"

    if index and not middle and not ring and pinky:
        return "摇滚"
    if not index and middle and not ring and not pinky:
        return "竖中指"
    if index and middle and ring and not pinky:
        return "三指"
    if thumb and index and middle and not ring and not pinky:
        return "爪子"
    if thumb and index and not middle and not ring and not pinky:
        return "捏"
    if not thumb and index and middle and ring and pinky:
        return "四指"

    return None


# ==================== 身体姿态识别 ====================

def recognize_pose(pose_landmarks):
    if pose_landmarks is None:
        return []

    lm = pose_landmarks.landmark
    actions = []

    L_SHOULDER, R_SHOULDER = 11, 12
    L_ELBOW, R_ELBOW = 13, 14
    L_WRIST, R_WRIST = 15, 16
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28

    left_up = lm[L_WRIST].y < lm[L_SHOULDER].y
    right_up = lm[R_WRIST].y < lm[R_SHOULDER].y
    if left_up and right_up:
        actions.append("双臂举起")
    elif left_up:
        actions.append("左手举起")
    elif right_up:
        actions.append("右手举起")

    hip_knee = (abs(lm[L_HIP].y - lm[L_KNEE].y) + abs(lm[R_HIP].y - lm[R_KNEE].y)) / 2
    if hip_knee < 0.1:
        actions.append("坐下")

    if (abs(lm[L_WRIST].y - lm[L_SHOULDER].y) < 0.08 and
        abs(lm[R_WRIST].y - lm[R_SHOULDER].y) < 0.08 and
        lm[L_WRIST].x < lm[L_SHOULDER].x and
        lm[R_WRIST].x > lm[R_SHOULDER].x):
        actions.append("张开双臂")

    hand_dist = math.sqrt((lm[L_WRIST].x - lm[R_WRIST].x) ** 2 + (lm[L_WRIST].y - lm[R_WRIST].y) ** 2)
    if hand_dist < 0.05 and lm[L_WRIST].y < lm[L_SHOULDER].y:
        actions.append("双手合十")

    if (abs(lm[L_WRIST].y - lm[L_HIP].y) < 0.08 and
        abs(lm[R_WRIST].y - lm[R_HIP].y) < 0.08 and
        abs(lm[L_WRIST].x - lm[L_HIP].x) < 0.15 and
        abs(lm[R_WRIST].x - lm[R_HIP].x) < 0.15):
        actions.append("双手叉腰")

    if lm[L_WRIST].y < lm[L_ELBOW].y < lm[L_SHOULDER].y and lm[L_ELBOW].x < lm[L_SHOULDER].x:
        actions.append("左臂弯曲")
    if lm[R_WRIST].y < lm[R_ELBOW].y < lm[R_SHOULDER].y and lm[R_ELBOW].x > lm[R_SHOULDER].x:
        actions.append("右臂弯曲")

    if lm[L_KNEE].y < lm[L_HIP].y:
        actions.append("左腿抬起")
    if lm[R_KNEE].y < lm[R_HIP].y:
        actions.append("右腿抬起")

    if abs(lm[L_KNEE].y - lm[R_KNEE].y) > 0.15:
        actions.append("弓步")

    if abs(lm[L_ANKLE].y - lm[R_ANKLE].y) > 0.1:
        actions.append("单脚站立")

    mid_shoulder_y = (lm[L_SHOULDER].y + lm[R_SHOULDER].y) / 2
    nose_y = lm[0].y
    if nose_y - mid_shoulder_y > 0.1:
        actions.append("鞠躬")

    return actions if actions else ["站立"]


# ==================== UI 绘制 ====================

def draw_rounded_rect(img, x, y, w, h, color, radius=15, alpha=0.7):
    overlay = img.copy()
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, -1)
    cv2.circle(overlay, (x + radius, y + radius), radius, color, -1)
    cv2.circle(overlay, (x + w - radius, y + radius), radius, color, -1)
    cv2.circle(overlay, (x + radius, y + h - radius), radius, color, -1)
    cv2.circle(overlay, (x + w - radius, y + h - radius), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_objects(frame, results, class_names):
    for r in results:
        x, y, w, h = r["box"]
        cn_name = COCO_CN.get(class_names[r["class_id"]], class_names[r["class_id"]])
        label = f"{cn_name} {r['confidence']:.0%}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), C_OBJ, 2)
        tw, th = get_text_size(label, FONT_18)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 10, y), C_OBJ, -1)
        frame = put_cn_text(frame, label, (x + 5, y - th - 5), FONT_18, (0, 0, 0))
    return frame


def draw_info_panel(frame, gestures, poses, fps, particle_count):
    _, w = frame.shape[:2]
    panel_w = 240
    panel_x = w - panel_w - 10

    draw_rounded_rect(frame, panel_x, 10, panel_w, 330, C_BG, radius=12, alpha=0.75)

    y = 35
    frame = put_cn_text(frame, "AI 智能视觉", (panel_x + 55, y), FONT_24, C_ACCENT)
    y += 35

    cv2.line(frame, (panel_x + 15, y), (panel_x + panel_w - 15, y), C_ACCENT, 1)
    y += 15

    frame = put_cn_text(frame, f"帧率: {fps:.0f} FPS", (panel_x + 15, y), FONT_16, C_TEXT)
    y += 24
    frame = put_cn_text(frame, f"粒子: {particle_count}", (panel_x + 15, y), FONT_16, C_DIM)
    y += 28

    # 手势
    frame = put_cn_text(frame, "[ 手势识别 ]", (panel_x + 15, y), FONT_18, C_GESTURE)
    y += 28
    if gestures:
        for g in gestures:
            frame = put_cn_text(frame, f"  {g}", (panel_x + 15, y), FONT_16, C_TEXT)
            y += 24
    else:
        frame = put_cn_text(frame, "  未检测到手势", (panel_x + 15, y), FONT_16, C_DIM)
        y += 24

    y += 10

    # 姿态
    frame = put_cn_text(frame, "[ 姿态识别 ]", (panel_x + 15, y), FONT_18, C_POSE)
    y += 28
    if poses:
        for p in poses:
            frame = put_cn_text(frame, f"  {p}", (panel_x + 15, y), FONT_16, C_TEXT)
            y += 24
    else:
        frame = put_cn_text(frame, "  未检测到姿态", (panel_x + 15, y), FONT_16, C_DIM)

    return frame


# ==================== 主程序 ====================

def main():
    print("加载 YOLO 模型...")
    net, class_names, output_layers = load_yolo()
    print(f"YOLO 加载完成，可识别 {len(class_names)} 类物体")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return

    print("摄像头已打开，按 'q' 退出")
    prev_time = time.time()
    particles = ParticleSystem(max_particles=800)
    prev_gesture: list[str | None] = [None, None]  # 跟踪上一帧手势，用于触发动画
    frame_count = 0

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands, mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取画面")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1

            # 色相随时间缓慢变化
            particles.set_hue((frame_count * 2) % 180)

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            # 1) YOLO 目标检测
            yolo_results = detect_objects(net, output_layers, frame)
            frame = draw_objects(frame, yolo_results, class_names)

            # 2) 手势识别 + 粒子生成
            hand_results = hands.process(rgb)
            gestures = []
            if hand_results.multi_hand_landmarks:
                for idx, hand_lms in enumerate(hand_results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
                    gesture = recognize_gesture(hand_lms)
                    if gesture:
                        gestures.append(gesture)

                    # 五指指尖生成粒子
                    finger_tips = [4, 8, 12, 16, 20]
                    for tip_id in finger_tips:
                        lx = int(hand_lms.landmark[tip_id].x * w)
                        ly = int(hand_lms.landmark[tip_id].y * h)
                        particles.spawn_trail(lx, ly, count=2)

                    # 手势切换时触发爆发效果
                    if idx < 2 and gesture != prev_gesture[idx] and gesture is not None:
                        palm_x = int(hand_lms.landmark[9].x * w)
                        palm_y = int(hand_lms.landmark[9].y * h)
                        particles.spawn_burst(palm_x, palm_y, count=25)

                    if idx < 2:
                        prev_gesture[idx] = gesture

            else:
                prev_gesture = [None, None]

            # 3) 身体姿态识别
            pose_results = pose.process(rgb)
            poses = []
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style(),
                )
                poses = recognize_pose(pose_results.pose_landmarks)

            # 4) 更新并绘制粒子
            particles.update()
            frame = particles.draw(frame)

            # 5) 信息面板
            frame = draw_info_panel(frame, gestures, poses, fps, len(particles.particles))

            cv2.imshow("AI Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
