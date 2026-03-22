import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import time
import os


TEAM_NAME = "КОМАНДА №5"
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "test_video.mp4")


if not os.path.exists(video_path):
    video_path = r"D:\KION\test_video.mp4" 

model = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture(video_path)

for f in ['incidents', 'smoking_incidents']:
    if not os.path.exists(os.path.join(script_dir, f)): 
        os.makedirs(os.path.join(script_dir, f))


root = tk.Tk()
root.title(f"KION - {TEAM_NAME}")
root.geometry("1250x820")
root.configure(bg="#0f172a")


smoke_timers = {}
last_save_time = 0
counts = {"fight": 0, "smoke": 0}
is_paused = False  


header = tk.Frame(root, bg="#1e293b", height=60)
header.pack(side="top", fill="x")
tk.Label(header, text=f"KION | {TEAM_NAME}", font=("Segoe UI", 18, "bold"), 
         bg="#1e293b", fg="#f8fafc").pack(pady=10)

# Видео
video_frame = tk.Frame(root, bg="#0f172a", bd=2)
video_frame.pack(side="left", padx=20, pady=10)
video_label = tk.Label(video_frame, bg="black", width=800, height=500)
video_label.pack()


ctrl_frame = tk.Frame(root, bg="#0f172a")
ctrl_frame.pack(side="bottom", fill="x", padx=20, pady=10)

def toggle_pause():
    global is_paused
    is_paused = not is_paused
    btn_pause.config(text="ПУСК" if is_paused else "ПАУЗА", bg="#f59e0b" if is_paused else "#6366f1")


btn_pause = tk.Canvas(ctrl_frame, width=140, height=40, bd=0, highlightthickness=0, cursor="hand2")
btn_pause.pack(side="left", padx=10)

def draw_pause_gradient(canvas):
    w = canvas.winfo_width()
    h = canvas.winfo_height()
    
    
    c1 = (16, 15, 46)   # #100F2E
    c2 = (96, 15, 64)   # #600F40
    
    for i in range(w):
        
        r = int(c1[0] + (c2[0] - c1[0]) * (i / w))
        g = int(c1[1] + (c2[1] - c1[1]) * (i / w))
        b = int(c1[2] + (c2[2] - c1[2]) * (i / w))
        color = f'#{r:02x}{g:02x}{b:02x}'
        canvas.create_line(i, 0, i, h, fill=color)
    
    
    canvas.create_text(w/2, h/2, text="ПАУЗА", fill="white", font=("Segoe UI", 12, "bold"))


btn_pause.bind("<Configure>", lambda e: draw_pause_gradient(btn_pause))
btn_pause.bind("<Button-1>", lambda e: toggle_pause())



right_panel = tk.Frame(root, bg="#1e293b", width=350)
right_panel.pack(side="right", fill="both", padx=10, pady=10)

tk.Label(right_panel, text="СТАТИСТИКА", font=("Segoe UI", 12, "bold"), bg="#1e293b", fg="#94a3b8").pack(pady=15)
fight_lbl = tk.Label(right_panel, text="ДРАКИ: 0", font=("Segoe UI", 14, "bold"), bg="#1e293b", fg="#ef4444")
fight_lbl.pack()
smoke_lbl = tk.Label(right_panel, text="КУРЕНИЕ: 0", font=("Segoe UI", 14, "bold"), bg="#1e293b", fg="#fbbf24")
smoke_lbl.pack()

log_box = scrolledtext.ScrolledText(right_panel, width=40, height=20, font=("Consolas", 10), bg="#0f172a", fg="#f1f5f9")
log_box.pack(padx=15, pady=20)

def open_incidents():
    path = os.path.join(script_dir, 'incidents')
    if os.path.exists(path): os.startfile(path)


btn_canvas = tk.Canvas(right_panel, height=45, bd=0, highlightthickness=0, cursor="hand2")
btn_canvas.pack(side="bottom", fill="x", padx=30, pady=20)

def draw_gradient(canvas):
    width = canvas.winfo_width()
    height = canvas.winfo_height()
    
    
    c1 = (16, 15, 46)   # #100F2E
    c2 = (96, 15, 64)   # #600F40
    
    for i in range(width):
        
        r = int(c1[0] + (c2[0] - c1[0]) * (i / width))
        g = int(c1[1] + (c2[1] - c1[1]) * (i / width))
        b = int(c1[2] + (c2[2] - c1[2]) * (i / width))
        color = f'#{r:02x}{g:02x}{b:02x}'
        canvas.create_line(i, 0, i, height, fill=color)
    
    
    canvas.create_text(width/2, height/2, text="ОТКРЫТЬ АРХИВ", 
                       fill="white", font=("Segoe UI", 11, "bold"))


btn_canvas.bind("<Configure>", lambda e: draw_gradient(btn_canvas))

btn_canvas.bind("<Button-1>", lambda e: open_incidents())



def process():
    global last_save_time, is_paused
    
    if is_paused:
        root.after(100, process) 
        return

    success, frame = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        root.after(10, process)
        return

    
    results = model(frame, verbose=False, conf=0.3)[0]
    annotated_frame = results.plot()
    
    people = []
    if results.keypoints is not None and len(results.keypoints.xy) > 0:
        people = results.keypoints.xy.cpu().numpy()

    # ДРАКА 
    if len(people) >= 2:
        d = np.linalg.norm(people[0][0] - people[1][0]) # Расстояние между носами
        if d < 100 and (time.time() - last_save_time > 5):
            counts["fight"] += 1
            fight_lbl.config(text=f"ДРАКИ: {counts['fight']}")
            log_box.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] ALERT: FIGHT!\n")
            cv2.imwrite(os.path.join(script_dir, "incidents", f"f_{int(time.time())}.jpg"), annotated_frame)
            last_save_time = time.time()

    
    img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2.resize(img, (800, 500)))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    
    root.after(10, process)


if __name__ == "__main__":
    process()
    root.mainloop()
    cap.release()
