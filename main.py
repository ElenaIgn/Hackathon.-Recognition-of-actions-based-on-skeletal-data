import cv2
from ultralytics import YOLO

#  Загружаем модель
model = YOLO('yolov8n-pose.pt') 


cap = cv2.VideoCapture(r"D:\ТГУ\Хакатон КИОН\yolo\test_video.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    
    results = model(frame, verbose=False)


    
    for r in results:
        # Рисуем скелет
        annotated_frame = r.plot()
        
       
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            # Получаем координаты (x, y) 
            kp = r.keypoints.xy[0].cpu().numpy()
            
            
            if len(kp) > 10:
                if kp[10][1] < kp[0][1] and kp[10][1] != 0:
                    cv2.putText(annotated_frame, "HAND UP!", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.imshow("YOLOv8 Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

