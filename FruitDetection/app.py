import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from inference import get_model
import supervision as sv

# Tải model từ Roboflow
model = get_model(model_id="fruit-detection-wgncl/1", api_key='Mo5T7f1BSKMqlEVsDWqO')

# Hàm xử lý và annotate ảnh
def detect_and_display(image_path):
    # Đọc ảnh bằng OpenCV
    image =  cv2.imread(image_path)

    # Inference từ  Roboflow
    results = model.infer(image)[0]
    detections = sv.Detections.from_inference(results)

    # Annotate ảnh
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Chuyển ảnh OpenCV sang định dạng hiển thị được trong tkinter
    image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tk = ImageTk.PhotoImage(image_pil)

    # Hiển thị ảnh lên canvas
    canvas.image = image_tk
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

# Hàm chọn file ảnh từ máy
def choose_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        detect_and_display(file_path)

# Tạo giao diện chính
root = tk.Tk()
root.title("Fruit Detection with Roboflow")
root.geometry("400x400")

# Nút chọn ảnh
btn = tk.Button(root, text="Chọn ảnh", command=choose_image, font=("Arial", 14))
btn.pack(pady=10)

# Canvas hiển thị ảnh
canvas = tk.Canvas(root, width=800, height=500)
canvas.pack()

root.mainloop()
