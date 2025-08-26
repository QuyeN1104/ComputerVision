import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import cv2
from collections import Counter
from ultralytics import YOLO

# ===== Cấu hình =====
MODEL_PATH = 'best_tomoto.pt'  # đổi lại nếu cần

# ===== Khởi tạo YOLO =====
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise SystemExit(f"Không thể load model tại '{MODEL_PATH}': {e}")

# ===== Ứng dụng =====
root = tk.Tk()
root.title("YOLO Object Detection")

# Mở full màn hình theo kiểu Windows (zoomed). Nếu muốn full thật sự: root.attributes('-fullscreen', True)
root.state('zoomed')

# Toggle fullscreen với F11, thoát với Esc (tuỳ chọn)
is_fullscreen = [False]
def toggle_fullscreen(event=None):
    is_fullscreen[0] = not is_fullscreen[0]
    root.attributes('-fullscreen', is_fullscreen[0])
def end_fullscreen(event=None):
    is_fullscreen[0] = False
    root.attributes('-fullscreen', False)
root.bind("<F11>", toggle_fullscreen)
root.bind("<Escape>", end_fullscreen)

# ===== Layout =====
top_bar = tk.Frame(root)
top_bar.pack(side='top', fill='x', pady=8)

content = tk.Frame(root)
content.pack(side='top', fill='both', expand=True)

left_panel = tk.Frame(content, bg='#111')
left_panel.pack(side='left', fill='both', expand=True, padx=(12, 6), pady=12)

right_panel = tk.Frame(content, bg='#111')
right_panel.pack(side='right', fill='both', expand=True, padx=(6, 12), pady=12)

# Nhãn tiêu đề cho mỗi panel
left_title = tk.Label(left_panel, text="Ảnh gốc", font=('Arial', 14, 'bold'), fg='white', bg='#111')
left_title.pack(anchor='w', padx=8, pady=(8,4))

right_title = tk.Label(right_panel, text="Ảnh dự đoán (YOLO)", font=('Arial', 14, 'bold'), fg='white', bg='#111')
right_title.pack(anchor='w', padx=8, pady=(8,4))

# Nơi hiển thị ảnh
original_label = tk.Label(left_panel, bg='#111')
original_label.pack(fill='both', expand=True, padx=8, pady=8)

predicted_label = tk.Label(right_panel, bg='#111')
predicted_label.pack(fill='both', expand=True, padx=8, pady=(8,4))

# Hiển thị số lượng đối tượng
count_label = tk.Label(right_panel, text="", font=('Arial', 12), fg='white', bg='#111', justify='left')
count_label.pack(anchor='w', padx=8, pady=(0,10))

# Biến lưu ảnh gốc & ảnh đã vẽ để resize lại khi cửa sổ thay đổi kích thước
state = {
    "original_pil": None,
    "pred_pil": None
}

def fit_to_label(pil_img: Image.Image, container: tk.Label) -> ImageTk.PhotoImage:
    """Co ảnh theo kích thước nhãn, giữ tỉ lệ."""
    if pil_img is None:
        return None
    # Lấy kích thước khả dụng của label
    w = container.winfo_width()
    h = container.winfo_height()
    # Nếu panel chưa đo được (mới khởi động), tạm đặt kích thước lớn
    if w < 10 or h < 10:
        w, h = 800, 600
    # Dùng ImageOps.contain để giữ tỉ lệ
    img = ImageOps.contain(pil_img, (w, h), method=Image.LANCZOS)
    return ImageTk.PhotoImage(img)

def render_images():
    """Vẽ lại ảnh theo kích thước panel hiện tại."""
    if state["original_pil"] is not None:
        photo = fit_to_label(state["original_pil"], original_label)
        original_label.config(image=photo)
        original_label.image = photo
    if state["pred_pil"] is not None:
        photo_pred = fit_to_label(state["pred_pil"], predicted_label)
        predicted_label.config(image=photo_pred)
        predicted_label.image = photo_pred

def on_resize(event):
    # Mỗi khi panel thay đổi kích thước -> vẽ lại
    root.after(10, render_images)

left_panel.bind("<Configure>", on_resize)
right_panel.bind("<Configure>", on_resize)

def select_image():
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    if not file_path:
        return

    try:
        # 1) Ảnh gốc (PIL)
        original = Image.open(file_path).convert("RGB")
        state["original_pil"] = original

        # 2) Chạy YOLO
        results = model(file_path)[0]

        # 3) Vẽ bbox trong bộ nhớ (BGR -> RGB -> PIL)
        annotated_bgr = results.plot()                   # numpy array (BGR)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        pred_pil = Image.fromarray(annotated_rgb)
        state["pred_pil"] = pred_pil

        # 4) Cập nhật ảnh hiển thị (auto scale theo panel)
        render_images()

        # 5) Đếm số lượng đối tượng + theo lớp
        total = len(results.boxes) if results.boxes is not None else 0
        if total > 0:
            cls_ids = results.boxes.cls.int().tolist()
            counts = Counter(cls_ids)
            # Tên lớp
            names = results.names if hasattr(results, "names") and results.names else model.names
            by_class = ", ".join(f"{names[c]}: {counts[c]}" for c in sorted(counts))
            count_label.config(text=f"Tổng số đối tượng: {total}\nTheo lớp: {by_class}")
        else:
            count_label.config(text="Tổng số đối tượng: 0")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể xử lý ảnh:\n{e}")

# Nút chọn ảnh
btn = tk.Button(top_bar, text="Chọn ảnh", command=select_image,
                font=('Arial', 14), bg='lightblue', activebackground='#7ec8e3')
btn.pack(side='left', padx=12)

root.mainloop()
