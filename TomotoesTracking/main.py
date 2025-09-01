import cv2
from collections import defaultdict
from ultralytics import YOLO

# ================== CẤU HÌNH ==================
VIDEO   = "tomato_3.MOV"          # đường dẫn video (.mp4/.MOV đều được)
MODEL   = "best.pt"         # hoặc yolov11n.pt / model của bạn
OUT     = "tracked.mp4"        # file xuất

# ==============================================

# -------- Vẽ box + nhãn: chỉ tên + conf, KHÔNG hiển thị ID ----------
def draw_boxes_no_id(frame, boxes, names):
    """
    Tự vẽ khung và nhãn "class conf" mà KHÔNG hiển thị track_id.
    """
    for b in boxes:
        # Lấy thông tin box
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cls  = int(b.cls[0])
        conf = float(b.conf[0])

        # Vẽ khung
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Nhãn: "class conf"
        label = f"{names[cls]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        th = th + 6
        # nền nhãn
        cv2.rectangle(frame, (x1, y1 - th), (x1 + tw + 2, y1), (0, 225, 255), -1)
        # chữ
        cv2.putText(frame, label, (x1 + 1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

# -------- Vẽ bảng thống kê UNIQUE (theo track_id) ----------
def draw_stats_panel(frame, unique_ids_by_cls, names, x=8, y=8):
    """
    Vẽ bảng 'Counts (unique): <class>: <số track_id đã thấy>'
    """
    lines = ["Counts :"]
    # sắp theo tên lớp cho dễ đọc (hoặc bỏ dòng dưới để giữ thứ tự dict)
    for cls in sorted(unique_ids_by_cls.keys()):
        lines.append(f"{names[cls]}: {len(unique_ids_by_cls[cls])}")

    pad = 30
    line_h = 50
    w = max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] for s in lines) + 2*pad
    h = line_h * len(lines) + 2*pad

    # Nền mờ
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # Text
    yy = y + pad + 16
    for s in lines:
        cv2.putText(frame, s, (x + pad, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        yy += line_h

def main():
    # ====== MỞ VIDEO ======
    cap = cv2.VideoCapture(VIDEO)
    assert cap.isOpened(), f"Không mở được video: {VIDEO}"

    # Lấy FPS gốc
    fps = cap.get(cv2.CAP_PROP_FPS)

    # ====== LOAD MODEL ======
    model = YOLO(MODEL)
    # names của lớp (tùy version Ultralytics lưu ở chỗ khác nhau)
    names = model.model.names if hasattr(model, "model") else model.names

    # ====== BỘ ĐẾM UNIQUE ======
    # { class_id: set(track_id duy nhất đã thấy) }
    unique_ids_by_cls = defaultdict(set)

    # ====== CHUẨN BỊ GHI VIDEO ======
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # ====== VÒNG LẶP TỪNG FRAME ======
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ====== YOLO TRACK ======
        res = model.track(
            frame,
            persist=True
        )

        r = res[0]
        boxes = r.boxes  # danh sách box đã NMS (và đã có track_id nếu tracker gán kịp)

        # ====== CẬP NHẬT ĐẾM UNIQUE THEO track_id ======
        # Chỉ thêm khi b.id != None (đã có ID từ tracker)
        for b in boxes:
            if b.id is not None:
                tid = int(b.id[0])
                cls = int(b.cls[0])
                unique_ids_by_cls[cls].add(tid)

        # ====== VẼ ANNOTATE (box + tên + conf, KHÔNG ID) ======
        draw_boxes_no_id(frame, boxes, names)

        # ====== VẼ BẢNG THỐNG KÊ UNIQUE ======
        draw_stats_panel(frame, unique_ids_by_cls, names)

        # ====== KHỞI TẠO WRITER LẦN ĐẦU (dựa theo kích thước frame hiện tại) ======
        if writer is None:
            h, w = frame.shape[:2]
            # ép số chẵn cho codec (tránh một số codec bị lỗi/crop khi w,h lẻ)

            writer = cv2.VideoWriter(OUT, fourcc, fps//4, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Không khởi tạo được VideoWriter: {OUT}")
            # (tùy thích) in log
            print(f"[INIT WRITER] src={w}x{h}, fps={fps}")
            writer.write(frame)
        else:
            writer.write(frame)

        # ====== XEM PREVIEW (không ảnh hưởng file ghi) ======
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
        display = cv2.resize(frame, (1080, 720))
        cv2.imshow("preview", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    # ====== GIẢI PHÓNG ======
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
