# ----lỆNH IMPORT-----
import cv2
from tkinter import *
from tkinter import filedialog
import numpy as np
from PIL import ImageTk, Image
from rembg import remove, new_session
from ultralytics import YOLO
import threading
import os

# --- Khởi tạo ---
frame_original = None
running = True
captured_image = None
yolo_results = None

# Chuẩn ảnh thẻ Việt Nam (4x6cm )
PASSPORT_W =590
PASSPORT_H =800

# Tỷ lệ khuôn mặt theo chuẩn VN: chiều cao khuôn mặt chiếm 70–80% ảnh
FACE_RATIO = 0.7
# khoảng trống từ đỉnh đầu đến viền trên (pixel)
TOP_MARGIN = 20 

# Tải YOLO
try:
    yolo_model = YOLO("yolov8n.pt")
except Exception as e:  
    print(f"Lỗi YOLO: {e}")
    exit()

#khởi tạo mô hình tách nền(u2net)
try:
    rembg_session = new_session("u2net")
except Exception as e:
    print(f"Lỗi rembg: {e}")
    exit()

FRAME_WIDTH = 480
FRAME_HEIGHT = 720
YOLO_FPS_LIMIT = 10
frame_counter = 0

# --- Hàm xử lý ---
def save_captured_image():
    global captured_image
    if captured_image is None:
        status_label.config(text="Chưa có ảnh để lưu")
        return
    filepath = filedialog.asksaveasfilename(defaultextension=".jpg",
                                             filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
    if filepath:
        cv2.imwrite(filepath, cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR))
        status_label.config(text=f"Đã lưu: {filepath}")

def show_captured_image(img):
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
    result_window = Toplevel(window)
    result_window.title("Kết quả ảnh thẻ")
    Label(result_window, image=imgtk).pack()
    result_window.imgtk = imgtk
    Button(result_window, text="Lưu ảnh", command=save_captured_image,font=("Segoe UI", 14), bg="#13e900").pack(pady=5)


def crop_passport(img, bbox):
    h, w, _ = img.shape
    x1, y1, x2, y2 = bbox
    face_h = y2 - y1
    face_w = x2 - x1

    # Kích thước chuẩn khung 3x4
    crop_h = int(face_h / 0.70) # mặt chiếm ~70% khung
    crop_w = int(crop_h * PASSPORT_W / PASSPORT_H)

    # Căn giữa theo chiều ngang
    cx = (x1 + x2) // 2
    crop_x1 = max(0, int(cx - crop_w // 4.2)) 
    crop_x2 = min(w, int(cx + crop_w // 4.2))  

    # Căn đỉnh đầu cách viền trên ~60 px
    crop_y1 = max(0, int(y1 - 60))  
    crop_y2 = min(h, int(crop_y1 + crop_h))  

    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
    if cropped.size == 0:
        return None
    return cv2.resize(cropped, (PASSPORT_W, PASSPORT_H))


def process_capture(bg_color):
    global frame_original, captured_image, yolo_results
    if frame_original is None or not yolo_results or not yolo_results.boxes:  # Kiểm tra điều kiện ban đầu
        status_label.config(text="Không tìm thấy khuôn mặt")
        return
    try: 
        status_label.config(text="Đang tách nền...")
        img_bgr = cv2.cvtColor(frame_original, cv2.COLOR_RGB2BGR)  # Chuẩn bị ảnh để xử lý
        _, buf = cv2.imencode(".png", img_bgr)
        img_bytes = buf.tobytes()

        # --- Tách nền bằng rembg ---
        img_no_bg_bytes = remove(img_bytes, session=rembg_session)
        img_no_bg_np = np.frombuffer(img_no_bg_bytes, np.uint8)
        img_no_bg = cv2.imdecode(img_no_bg_np, cv2.IMREAD_UNCHANGED)

        if img_no_bg.shape[-1] != 4:  # Kiểm tra có kênh alpha hay không
            status_label.config(text="Lỗi tách nền")
            return

        
        # --- Xử lý viền nhẹ & mịn (ưu tiên ảnh thẻ) ---

        alpha = img_no_bg[:, :, 3]

        # Chuẩn hóa alpha
        alpha = cv2.normalize(alpha, None, 0, 255, cv2.NORM_MINMAX)

        # Kernel nhỏ – không phá chi tiết
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Chỉ loại nhiễu rất nhỏ
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=1)

        # Blur RẤT NHẸ (chỉ đủ mịn viền)
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

        # Chuẩn hóa về [0..1]
        alpha = alpha.astype(np.float32) / 255.0


        # --- Feather chỉ vùng viền ngoài (KHÔNG ăn vào mặt) ---
        edge = cv2.Canny((alpha * 255).astype(np.uint8), 50, 150)
        edge = cv2.dilate(edge, kernel, iterations=1)

        edge_mask = cv2.GaussianBlur(edge.astype(np.float32), (7, 7), 0)
        edge_mask = np.clip(edge_mask / 255.0, 0, 1)

        # Chỉ làm mềm alpha tại viền
        alpha = alpha * (1 - 0.4 * edge_mask)


        # --- Khử viền ám màu RẤT NHẸ ---
        fg = img_no_bg[:, :, :3].astype(np.float32)

        for c in range(3):
            fg[:, :, c] = np.clip(
                fg[:, :, c] - (1 - alpha) * 8,  # GIẢM MẠNH so với bản cũ
                0, 255
            )


        # --- Ghép nền ---
        bg = np.full(fg.shape, bg_color, dtype=np.float32)
        combined = fg * alpha[..., None] + bg * (1 - alpha[..., None])
        combined = combined.astype(np.uint8)

        combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

        # --- Lấy bounding box lớn nhất từ YOLO ---
        bboxes = yolo_results.boxes.xyxy.cpu().numpy().astype(int)
        person_box = max(bboxes, key=lambda box: (box[2]-box[0]) * (box[3]-box[1]))

        # --- Crop ảnh kiểu ảnh thẻ ---
        final_img = crop_passport(combined, person_box)
        if final_img is None:
            status_label.config(text="Lỗi crop")
            return

        captured_image = final_img
        show_captured_image(captured_image)
        status_label.config(text="Xong!")
    except Exception as e:
        status_label.config(text=f"Lỗi: {e}")


#Nền màu trắng
def capture_white():
    threading.Thread(target=process_capture, args=((255,255,255),)).start()
#Nền màu xanh
def capture_blue():
    threading.Thread(target=process_capture, args=((255,125,0),)).start()

def show_frame():
    global frame_original, yolo_results, frame_counter
    if not running:
        return
    ret, raw_frame = camera.read()
    if not ret:
        window.after(10, show_frame)
        return
    raw_frame = cv2.resize(raw_frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame_original = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
    frame_disp = frame_original.copy()
    if frame_counter % YOLO_FPS_LIMIT == 0:
        results = yolo_model(frame_disp, classes=[0], verbose=False)
        yolo_results = results[0] if results and len(results[0].boxes) > 0 else None
    if yolo_results and yolo_results.boxes:
        bboxes = yolo_results.boxes.xyxy.cpu().numpy().astype(int)
        main_box = max(bboxes, key=lambda box:(box[2]-box[0])*(box[3]-box[1]))
        x1,y1,x2,y2 = main_box
        cv2.rectangle(frame_disp,(x1,y1),(x2,y2),(0,255,0),2)
        reminder_label.config(text="✅ Đã phát hiện người", fg="green")
    else:
        reminder_label.config(text="📍 Đứng vào giữa khung hình", fg="red")
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame_disp))
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    frame_counter += 1
    window.after(20, show_frame)

#Tắt phần mềm 
def on_closing():
    global running 
    running = False
    camera.release()
    window.destroy()

# --- GUI ---
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera.isOpened():
    print("Không mở được camera")
    exit()

window = Tk()
window.title("📸 Phần mềm ảnh thẻ thông minh")
window.geometry("1920x1080")
window.configure(bg="#f0f0f0")
window.protocol("WM_DELETE_WINDOW", on_closing)

header = Label(window, text="📸 Phần mềm ảnh thẻ thông minh", font=("Segoe UI", 24, "bold"), bg="#ffffff")
header.pack(pady=10)

try: logo_img = ImageTk.PhotoImage(Image.open("images.jfif").resize((80, 80))); Label(window, image=logo_img, bg="#f0f0f0").place(x=20, y=10)
except: pass

main_frame = Frame(window, bg="#9B9B9B")
main_frame.pack(fill=BOTH, expand=True, padx=20, pady=10)

video_label = Label(main_frame, bg="#fff")
video_label.pack(side=LEFT, expand=True, fill=BOTH)

right_panel = Frame(main_frame, bg="#DA8E00", width=300)
right_panel.pack(side=RIGHT, fill=Y, padx=20)
right_panel.pack_propagate(False)

reminder_label = Label(right_panel, text="📍 Khởi tạo...", font=("Segoe UI", 14), bg="#f0f0f0", fg="red")
reminder_label.pack(pady=20)

Button(right_panel, text="📷 Chụp nền trắng", command=capture_white, font=("Segoe UI",14), bg="white").pack(pady=10, fill=X)
Button(right_panel, text="📷 Chụp nền xanh", command=capture_blue, font=("Segoe UI",14), bg="#3498db", fg="white").pack(pady=10, fill=X)
Button(right_panel, text="❌ Thoát", command=on_closing, font=("Segoe UI",14), bg="#333", fg="white").pack(pady=10, fill=X)

status_label = Label(window, text="Trạng thái: Sẵn sàng", font=("Segoe UI", 11), bg="#f0f0f0", fg="green")
status_label.pack(side=BOTTOM, pady=5)

show_frame()
window.mainloop()
