import cv2
import numpy as np
from ultralytics import YOLO
import os
import time


if not os.path.exists('incidents'):
    os.makedirs('incidents')

model = YOLO('yolov8n-pose.pt')
input_video = r"D:\ТГУ\Хакатон КИОН\yolo\test_video_2.mp4" 
cap = cv2.VideoCapture(input_video)

# Параметры записи
width, height = int(cap.get(3)), int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('KION_FINAL_REPORT.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

def get_dist(p1, p2):
    return np.linalg.norm(p1 - p2)

prev_knee_y = 0
last_save_time = 0  # Время последнего скриншота

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model(frame, verbose=False, conf=0.3)
    annotated_frame = frame.copy()
    
    people = []
    for r in results:
        annotated_frame = r.plot() 
        if r.keypoints is not None:
            kpts = r.keypoints.xy.cpu().numpy()
            for person_kpts in kpts:
                if len(person_kpts) > 15:
                    people.append(person_kpts)

    action = "Waiting..."
    color = (255, 255, 255)

    # --- Драка ---
    if len(people) >= 2:
        p1, p2 = people[0], people[1]
        d_hands = get_dist(p1[10], p2[10])      
        d_shoulders = get_dist(p1[5], p2[5])    
        d_fight = get_dist(p1[10], p2[0]) # Кулак к носу

        if d_fight < 50:
            action, color = "FIGHT!", (0, 0, 255)
            # СОХРАНЕНИЕ СКРИНШОТА ПРИ ДРАКЕ
            current_time = time.time()
            if current_time - last_save_time > 2: # Пауза 2 сек
                img_name = f"incidents/fight_{int(current_time)}.jpg"
                cv2.imwrite(img_name, annotated_frame)
                last_save_time = current_time
                print(f"[ALERT] Инцидент сохранен: {img_name}")

        elif d_shoulders < 80:
            action, color = "Hugging", (255, 0, 255)
        elif d_hands < 60:
            action, color = "Handshake", (0, 255, 0)

    # --- Танец, бег ---
    elif len(people) == 1:
        p = people[0]
        # 10-запястье, 6-плечо, 14-колено
        if p[10][1] < p[6][1] and p[10][1] != 0:
            action, color = "Dancing", (255, 255, 0)
        elif prev_knee_y != 0 and abs(p[14][1] - prev_knee_y) > 15:
            action, color = "Running", (0, 255, 255)
        prev_knee_y = p[14][1]

    
    cv2.rectangle(annotated_frame, (20, 20), (480, 100), (0, 0, 0), -1)
    cv2.putText(annotated_frame, f"ACTION: {action}", (40, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

    cv2.imshow("KION AI System", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
out.release()
cv2.destroyAllWindows()

