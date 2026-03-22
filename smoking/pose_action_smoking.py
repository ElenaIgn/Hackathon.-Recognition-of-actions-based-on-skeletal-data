import cv2
import numpy as np
from ultralytics import YOLO
import time
import os


folder = 'smoking_incidents'
if not os.path.exists(folder):
    os.makedirs(folder)

model = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture(r"D:\ТГУ\Хакатон КИОН\smo\test_video_3.mp4")

timers = {} 
last_save = 0  

def get_dist(p1, p2):
    return np.linalg.norm(p1 - p2)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model(frame, verbose=False, conf=0.3)
    annotated_frame = frame.copy()
    
    for r in results:
        if r.keypoints is None: continue
        kpts = r.keypoints.xy.cpu().numpy()
        
        for i, p in enumerate(kpts):
            nose, l_eye, r_eye, l_wrist, r_wrist = p[0], p[1], p[2], p[9], p[10]

            if np.all(nose == 0): continue 

            
            eye_dist = get_dist(l_eye, r_eye) if np.any(l_eye) and np.any(r_eye) else 30
            radius = int(eye_dist * 4.0) 
            
            
            d_l = get_dist(l_wrist, nose) if np.any(l_wrist) else 999
            d_r = get_dist(r_wrist, nose) if np.any(r_wrist) else 999
            
            hand_in_zone = (d_l < radius or d_r < radius)

            
            cv2.circle(annotated_frame, tuple(nose.astype(int)), radius, (0, 255, 0), 2)

            if hand_in_zone:
                if i not in timers: timers[i] = time.time()
                elapsed = time.time() - timers[i]
                
                
                cv2.circle(annotated_frame, tuple(nose.astype(int)), radius, (0, 0, 255), 3)
                cv2.putText(annotated_frame, f"Checking... {elapsed:.1f}s", (int(nose[0])-40, int(nose[1])-radius-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                
                if elapsed > 1.5: 
                    cv2.putText(annotated_frame, "!!! SMOKING !!!", (int(nose[0])-80, int(nose[1])-radius-40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                    
                    
                    current_time = time.time()
                    if current_time - last_save > 3:
                        filename = f"{folder}/smoke_{int(current_time)}.jpg"
                        
                        success_save = cv2.imwrite(filename, annotated_frame)
                        if success_save:
                            print(f"[SUCCESS] Кадр сохранен: {filename}")
                            last_save = current_time
                        else:
                            print("[ERROR] Не удалось записать файл. Проверьте права доступа.")
            else:
                if i in timers: del timers[i]

    cv2.imshow("KION Smoke Recorder", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()
