
import cv2
import math
from collections import deque, defaultdict
from ultralytics import YOLO

# ===================== CẤU HÌNH =====================
VIDEO_IN   = "test1.mp4"
VIDEO_OUT  = "tracked_perclass_dir_speed.mp4"
MODEL_PATH = "best.pt"          # đổi sang best.pt nếu là model đã train
TRACKER    = "bytetrack.yaml"

# Line đếm (2 điểm). Ví dụ: đường ngang y=485
LINE_A = (0, 485)
LINE_B = (1273, 485)

# (Tuỳ chọn) lọc lớp. Với COCO: car=2, bus=5, truck=7, motorbike=3, bicycle=1
CLASSES_FILTER = None        # None nếu muốn tất cả

# Ngưỡng detect/NMS
CONF = 0.35
IOU  = 0.7

# Hiển thị realtime
SHOW = True

# ==== Hiệu chỉnh tốc độ ====
# mét/pixel: bạn phải đo từ vật chuẩn trong khung hình (vd vạch kẻ 3.0 m ~ 60 px => 3/60 = 0.05)
METERS_PER_PIXEL = 0.05
# Cửa sổ thời gian (giây) để tính speed (mượt hơn khi tăng, nhưng phản ứng chậm lại)
SPEED_WINDOW_SEC = 0.6
# =====================================================


def side_of_line(A, B, P):
    # >0: bên trái, <0: bên phải (theo hướng A->B), =0: trên đường
    return (B[0]-A[0])*(P[1]-A[1]) - (B[1]-A[1])*(P[0]-A[0])


def compute_speed_kmh(track_hist: deque, fps: float, meters_per_pixel: float, window_sec: float):
    """track_hist: deque[(frame_idx, cx, cy)]"""
    if len(track_hist) < 2 or fps <= 0:
        return None
    cur_f, cur_x, cur_y = track_hist[-1]

    min_frames = max(1, int(window_sec * fps))
    # tìm điểm cũ nhất đủ xa trong cửa sổ
    old_idx = None
    for i in range(len(track_hist) - 2, -1, -1):
        f_i, _, _ = track_hist[i]
        if cur_f - f_i >= min_frames:
            old_idx = i
            break
    if old_idx is None:
        f_i, x_i, y_i = track_hist[0]
    else:
        f_i, x_i, y_i = track_hist[old_idx]

    dt = (cur_f - f_i) / fps
    if dt <= 0:
        return None
    dist_px = math.hypot(cur_x - x_i, cur_y - y_i)
    speed_ms = (dist_px * meters_per_pixel) / dt
    return speed_ms * 3.6


def get_class_name(names, cid):
    if isinstance(names, dict):
        return names.get(cid, str(cid))
    if isinstance(names, (list, tuple)):
        return names[cid] if 0 <= cid < len(names) else str(cid)
    return str(cid)


def draw_speed_label(frame, x, y, text):
    """Vẽ chữ to, đậm, đỏ có viền"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.15
    # outline đen dày
    cv2.putText(frame, text, (int(x), int(y)), font, scale, (0, 0, 0), 5, cv2.LINE_AA)
    # đỏ đậm
    cv2.putText(frame, text, (int(x), int(y)), font, scale, (0, 0, 255), 3, cv2.LINE_AA)


def draw_counts_panel(frame, total_A2B, total_B2A, per_class_A2B, per_class_B2A, x=20, y=40):
    # Tổng theo hướng
    cv2.putText(frame, f"A->B: {total_A2B}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 200, 255), 2)
    cv2.putText(frame, f"B->A: {total_B2A}", (x, y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 200, 255), 2)

    # Bảng theo từng class
    y0 = y + 75
    cv2.putText(frame, "Per-class:", (x, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    y0 += 30
    # Hiển thị tối đa ~12 dòng
    # Gom tất cả class xuất hiện trong 2 dict để hiện đủ
    all_classes = set(per_class_A2B.keys()) | set(per_class_B2A.keys())
    rows = []
    for cname in all_classes:
        rows.append((cname, per_class_A2B.get(cname, 0), per_class_B2A.get(cname, 0)))
    # sắp xếp theo tổng giảm dần
    rows.sort(key=lambda t: -(t[1] + t[2]))
    for i, (cname, a2b, b2a) in enumerate(rows[:12]):
        cv2.putText(frame, f"{cname}: A->B={a2b} | B->A={b2a}",
                    (x, y0 + i*28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)


def main():
    # Lấy thông số video
    cap = cv2.VideoCapture(VIDEO_IN)
    assert cap.isOpened(), f"Không mở được video: {VIDEO_IN}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 1e-3:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    cap.release()

    wait_ms = max(1, int(1000 / fps))

    # VideoWriter
    vw = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert vw.isOpened(), "Không mở được VideoWriter; thử đổi fourcc/đuôi file."

    # Model
    model = YOLO(MODEL_PATH)
    names = model.names

    # Trạng thái line-crossing
    last_side = {}  # tid -> side
    total_A2B = 0
    total_B2A = 0

    # Đếm RIÊNG cho từng class theo từng hướng
    per_class_A2B = defaultdict(int)
    per_class_B2A = defaultdict(int)

    # Lịch sử vị trí cho tính tốc độ (theo ID)
    hist_maxlen = int(fps * max(1.0, SPEED_WINDOW_SEC * 2.0))
    track_hist = defaultdict(lambda: deque(maxlen=hist_maxlen))

    if SHOW:
        cv2.namedWindow("Track/Count/Speed", cv2.WINDOW_NORMAL)

    frame_idx = 0
    try:
        results_gen = model.track(
            source=VIDEO_IN,
            stream=True,
            persist=True,
            tracker=TRACKER,
            conf=CONF,
            iou=IOU,
            classes=CLASSES_FILTER  # None hoặc list id lớp
        )

        for r in results_gen:
            frame = r.plot()

            # Vẽ line
            cv2.line(frame, LINE_A, LINE_B, (0, 255, 255), 2)
            cv2.putText(frame, "Count line",
                        ((LINE_A[0]+LINE_B[0])//2, (LINE_A[1]+LINE_B[1])//2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if r.boxes is not None and len(r.boxes) > 0 and r.boxes.id is not None:
                ids = r.boxes.id.int().tolist()
                xyxy = r.boxes.xyxy.tolist()
                cls  = r.boxes.cls.int().tolist()

                for i, tid in enumerate(ids):
                    x1, y1, x2, y2 = xyxy[i]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    cid = cls[i]
                    cname = get_class_name(names, cid)

                    # cập nhật lịch sử cho tính tốc độ
                    track_hist[tid].append((frame_idx, cx, cy))
                    speed_kmh = compute_speed_kmh(track_hist[tid], fps, METERS_PER_PIXEL, SPEED_WINDOW_SEC)

                    # NHÃN: chỉ TÊN + TỐC ĐỘ (không hiển thị ID)
                    if speed_kmh is not None:
                        text = f"{cname} | {speed_kmh:.1f} km/h"
                    else:
                        text = f"{cname}"
                    # vẽ ở góc trên trái bbox
                    draw_speed_label(frame, x1, max(20, y1 - 8), text)

                    # Đếm crossing qua line theo hướng
                    now_side = side_of_line(LINE_A, LINE_B, (cx, cy))
                    prev_side = last_side.get(tid)
                    if prev_side is None:
                        last_side[tid] = now_side
                    else:
                        if prev_side * now_side < 0:  # đổi dấu => qua line
                            if (prev_side < 0 and now_side > 0):
                                total_A2B += 1
                                per_class_A2B[cname] += 1
                            else:
                                total_B2A += 1
                                per_class_B2A[cname] += 1
                        last_side[tid] = now_side

            # Panel tổng + per-class
            draw_counts_panel(frame, total_A2B, total_B2A, per_class_A2B, per_class_B2A, x=20, y=40)

            # Ghi/hiển thị
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            vw.write(frame)

            if SHOW:
                cv2.imshow("Track/Count/Speed", frame)
                if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                    break

            frame_idx += 1

    finally:
        vw.release()
        if SHOW:
            cv2.destroyAllWindows()

    print(f"Đã lưu video: {VIDEO_OUT}")


if __name__ == "__main__":
    main()
