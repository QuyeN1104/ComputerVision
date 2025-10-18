import cv2
import csv
import numpy as np
from collections import defaultdict, Counter
from ultralytics import YOLO

# ================== CẤU HÌNH ==================
MODEL     = "best.pt"          # segmentation model (*-seg.pt)
SOURCE    = "test.mp4"         # "0" cho webcam
OUT_MP4   = "counting.mp4"
OUT_CSV   = "best_pixels.csv"  # file CSV xuất ra cuối cùng
CONF      = 0.25
IOU       = 0.45
IMG_SZ    = 640

USE_TRACK = True
TRACKER   = "bytetrack.yaml"   # đường dẫn hợp lệ tới bytetrack.yaml

# Làm mượt pixel bằng cách gom theo "bins" để trị số không dao động lặt vặt
PIXEL_BIN = 50                 # ví dụ 50; tăng lên -> ít nhảy hơn
SHOW_BEST_PIXEL_ONLY = True    # True: hiển thị mode-so-far, False: hiển thị pixel hiện tại

# Danh sách ZONE không chồng lấp: mỗi phần tử có name và points (polygon)
ZONES = [
    {"name": "RowA", "points": [[27, 927], [432, 927], [425, 1011], [24, 1008]]},
    {"name": "RowB", "points": [[906, 918], [906, 1018], [434, 1020], [441, 925]]},
    {"name": "RowC", "points": [[1330, 896], [1330, 1013], [916, 1022], [918, 915]]},
    {"name": "RowD", "points": [[1340, 899], [1340, 1020], [1781, 1013], [1779, 887]]},
]

# ================== HÀM PHỤ TRỢ ==================
def draw_label(img, x, y, text, fg=(255,255,255), bg=(0,80,180)):
    """Vẽ label và khung cho vật thể"""
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 5
    x2, y2 = x + tw + 2*pad, y
    y1 = max(0, y - th - 2*pad)
    cv2.rectangle(img, (x, y1), (x2, y), bg, -1)
    cv2.putText(img, text, (x + pad, y - pad), font, scale, fg, thick, cv2.LINE_AA)

def draw_zones(img, zones, color=(0, 220, 255)):
    """Vẽ polygon và nhãn Zone với tránh chồng chéo chữ."""
    pts = np.array(zone_pts, np.int32).reshape((-1,1,2))
    cv2.polylines(img, [pts], True, color, 2)
    cv2.putText(img, "ZONE", (pts[0,0,0], max(0, pts[0,0,1]-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

def centroid_of_box(x1, y1, x2, y2):
    """Tìm trọng tâm của một vật thể"""
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def which_zone(cx, cy, zones):
    """Trả về index zone chứa (cx,cy), hoặc -1 nếu không thuộc zone nào."""
    p = (float(cx), float(cy))
    for idx, z in enumerate(zones):
        pts = np.array(z["points"], np.int32)
        if cv2.pointPolygonTest(pts, p, False) >= 0:
            return idx
    return -1

def mask_up_from_result(i, r, h, w):
    """Lấy mask nhị phân (h,w) cho object i (ưu tiên r.masks.xy, fallback r.masks.data resize)."""
    if getattr(r, "masks", None) is None or r.masks is None:
        return None
    if getattr(r.masks, "xy", None) is not None:
        seg = r.masks.xy[i]
        m = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(seg, dtype=np.int32).reshape(-1,1,2)
        cv2.fillPoly(m, [pts], 1)
        return m
    return None

def overlay_mask_color(dst_bgr, mask01, color_bgr=(0, 0, 255), alpha=0.45):
    """Vẽ mask lên hình"""
    if mask01 is None:
        return
    m = mask01.astype(bool)
    if not m.any():
        return
    overlay = dst_bgr.copy()
    overlay[m] = (overlay[m]*(1-alpha) + np.array(color_bgr)*alpha).astype(overlay.dtype)
    dst_bgr[:] = overlay

def bin_value(v, bin_size):
    """Gom v vào bin: 0..bin_size-1 -> 0, bin_size..2*bin_size-1 -> bin_size, ..."""
    if bin_size <= 1:
        return int(v)
    return int(v // bin_size) * bin_size

# ================== MAIN ==================
def main():
    cap = cv2.VideoCapture(0 if str(SOURCE).isdigit() else SOURCE)
    assert cap.isOpened(), f"Không mở được nguồn: {SOURCE}"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    model = YOLO(MODEL)
    writer = cv2.VideoWriter(OUT_MP4, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
    assert writer.isOpened(), f"Không mở được file xuất: {OUT_MP4}"

    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

    # Bộ đếm & mapping theo zone:
    zone_counters = {z["name"]: 0 for z in ZONES}
    id_map = {}  # key: (zone_name, tracker_id) -> display_id string, vd "RowA/1"

    # Thống kê pixel: histogram (bin -> count), và "best" đang dẫn đầu cho hiển thị
    pixel_hists = defaultdict(Counter)   # (zone_name, tracker_id) -> Counter({bin_pix: freq, ...})
    best_pix_bin = {}                    # (zone_name, tracker_id) -> bin pixel tốt nhất (mode-so-far)
    best_pix_cnt = {}                    # (zone_name, tracker_id) -> tần suất cao nhất

    names = model.names if hasattr(model, "names") else {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Inference / Tracking
        if USE_TRACK:
            results = model.track(frame, conf=CONF, iou=IOU, imgsz=IMG_SZ,
                                  persist=True, tracker=TRACKER, verbose=False)
        else:
            results = model.predict(frame, conf=CONF, iou=IOU, imgsz=IMG_SZ, verbose=False)
        r = results[0]

        annotated = frame.copy()
        # draw_zones(annotated, ZONES)

        # Lấy boxes + id + cls + conf
        if getattr(r, "boxes", None) is not None and r.boxes is not None and r.boxes.xyxy is not None:
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else []
            clses = r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else []
            ids   = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else None
        else:
            xyxy, confs, clses, ids = [], [], [], None

        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            cx, cy = centroid_of_box(x1, y1, x2, y2)
            z_idx = which_zone(cx, cy, ZONES)

            pix_current = None
            pix_to_show = None
            display_id = ""
            in_zone = z_idx != -1

            # # Vẽ bbox
            # color = (0, 180, 0) if in_zone else (160, 160, 160)
            # cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = names.get(clses[i], str(clses[i])) if len(clses) > i else "obj"
            conf  = confs[i] if len(confs) > i else 0.0

            key = None
            if in_zone and ids is not None and len(ids) == len(xyxy):

                # Vẽ bbox
                color = (0, 180, 0) if in_zone else (160, 160, 160)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                zone_name = ZONES[z_idx]["name"]
                tracker_id = int(ids[i])
                key = (zone_name, tracker_id)

                # mapping tracker id -> display id dạng "Zone/num"
                if key not in id_map:
                    zone_counters[zone_name] += 1
                    id_map[key] = f"{zone_name}/{zone_counters[zone_name]}"
                display_id = id_map[key]

                # Tính pixel mask chỉ khi centroid còn trong zone
                m_up = mask_up_from_result(i, r, h, w)
                if m_up is not None:
                    pix_current = int(m_up.sum())
                    overlay_mask_color(annotated, m_up, (0, 0, 255), alpha=0.45)

                    # Cập nhật histogram theo bin để lấy mode
                    b = bin_value(pix_current, PIXEL_BIN)
                    pixel_hists[key][b] += 1
                    freq = pixel_hists[key][b]
                    # cập nhật best nếu cần
                    if (key not in best_pix_cnt) or (freq > best_pix_cnt[key]) or \
                       (freq == best_pix_cnt[key] and b < best_pix_bin[key]):  # tie-break: chọn bin nhỏ hơn
                        best_pix_cnt[key] = freq
                        best_pix_bin[key] = b

                # Chọn giá trị hiển thị: mode-so-far hoặc giá trị hiện tại
                if SHOW_BEST_PIXEL_ONLY and key in best_pix_bin:
                    pix_to_show = best_pix_bin[key]
                else:
                    pix_to_show = pix_current

                # Vẽ label
                text = f"{display_id} {label} {conf:.2f}".strip()
                if pix_to_show is not None:
                    text += f" | pix:{pix_to_show}"
                draw_label(annotated, x1, y1, text,
                           fg=(255,255,255),
                           bg=(0,120,0) if in_zone else (70,70,70))

                # centroid
                # cv2.circle(annotated, (cx, cy), 4, (0,0,255) if in_zone else (120,120,120), -1)

        cv2.imshow("Mask", annotated)
        writer.write(annotated)

        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
            break

    # ===== Xuất CSV: Hàng(Zone), Id (display_id), Pixel tốt nhất =====
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["Hàng(Zone)", "Id", "Pixel tốt nhất"])
        for key, best_bin in best_pix_bin.items():
            zone_name, tracker_id = key
            disp_id = id_map.get(key, f"{zone_name}/{tracker_id if tracker_id is not None else 'NA'}")
            wcsv.writerow([zone_name, disp_id, best_bin])

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"✅ Done. Saved video: {OUT_MP4}")
    print(f"✅ Saved CSV: {OUT_CSV}")

if __name__ == "__main__":
    main()
