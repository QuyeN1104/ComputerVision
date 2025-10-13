"""
1. Tạo mô hình YOLO
2. Duyệt qua từng frame của video:
3. Lấy được từng box/mask của mỗi vật thể trong frame
4. Kiểm tra xem trong các box đó box nào nằm trong vùng cần tính toán
5. Tính số pixel bằng cách chuyển về ma trận 0/1 (1 là nằm trong mask)
6. Tạo các annotated frame
7. dùng writer để ghi thành video
"""
import cv2
import numpy as np
from ultralytics import YOLO

# ================== CẤU HÌNH ==================
MODEL = 'best.pt'
TRACKER='bytetrack.yaml'
IOU=0.3
CONF = 0.5
SOURCE = 'test.mp4'
IMG_SZ = 640
OUT_MP4 = 'out_centroid_zone.mp4'
USE_TRACK = True


ZONE_POINTS = [[1, 322 +350], [1919, 322+350], [1912, 386+350], [-2, 386+350]]

# ================== HÀM PHỤ TRỢ ==================
def pixel_into_meters(num_pixels, conversion_factor = 0.5 / 13203):
    """
    :param num_pixels: number of pixels to convert
    :param conversion_factor: coversion_factor 1 pixel -> 0.5/13203 (m2)
    :return: number of squaremeter
    """
    return num_pixels * conversion_factor


def put_box_label(img, x1, y1, text, fg=(255,255,255), bg=(0,80,180)):
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 5
    cv2.rectangle(img, (x1, max(0, y1 - th - 2*pad)),
                  (x1 + tw + 2*pad, y1), bg, -1)
    cv2.putText(img, text, (x1 + pad, y1 - pad),
                font, scale, fg, thick, cv2.LINE_AA)

def draw_zone(img, zone_pts, color=(0, 220, 255)):
    pts = np.array(zone_pts, np.int32).reshape((-1,1,2))
    cv2.polylines(img, [pts], True, color, 2)
    cv2.putText(img, "ZONE", (pts[0,0,0], max(0, pts[0,0,1]-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def centroid_of_box(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def point_in_polygon(cx, cy, zone_pts):
    return cv2.pointPolygonTest(np.array(zone_pts, np.int32),
                                (float(cx), float(cy)), False) >= 0

def mask_up_from_result(i, r, h, w):
    """Lấy mask nhị phân (h,w) cho object i"""
    if getattr(r, "masks", None) is None or r.masks is None:
        return None
    if getattr(r.masks, "xy", None) is not None:
        seg = r.masks.xy[i]
        m = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(seg, dtype=np.int32).reshape(-1,1,2)
        cv2.fillPoly(m, [pts], 1)
        return m
    if getattr(r.masks, "data", None) is not None:
        m_small = (r.masks.data[i].detach().cpu().numpy() > 0.5).astype(np.uint8)
        return cv2.resize(m_small, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    return None

# ================== MAIN ==================
def main():
    cap = cv2.VideoCapture(0 if str(SOURCE).isdigit() else SOURCE)
    assert cap.isOpened(), f"Không mở được nguồn: {SOURCE}"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    model = YOLO(MODEL)
    out = cv2.VideoWriter(OUT_MP4, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if USE_TRACK:
            results = model.track(frame, conf=CONF, iou=IOU, imgsz=IMG_SZ,
                                  persist=True, tracker=TRACKER)
        else:
            results = model.predict(frame, conf=CONF, iou=IOU, imgsz=IMG_SZ, verbose=False)

        r = results[0]
        # print(r)

        annotated = frame.copy()
        draw_zone(annotated, ZONE_POINTS)

        # Lấy các thuộc tính xyxy, confs, clses, ids, names
        if getattr(r, "boxes", None) is not None and r.boxes is not None and r.boxes.xyxy is not None:
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else []
            clses = r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else []
            ids   = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else None
        else:
            xyxy, confs, clses, ids = [], [], [], None

        names = model.names if hasattr(model, "names") else {}

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = box
            cx, cy = centroid_of_box(x1, y1, x2, y2)

            in_zone = point_in_polygon(cx, cy, ZONE_POINTS)
            pix = None

            if in_zone:
                m_up = mask_up_from_result(i, r, h, w)
                if m_up is not None:
                    pix = int(m_up.sum())
                    color = (0, 0, 255)
                    alpha = 0.45
                    m3 = m_up.astype(bool)
                    overlay = annotated.copy()
                    overlay[m3] = (overlay[m3]*(1-alpha) + np.array(color)*alpha).astype(overlay.dtype)
                    annotated = overlay

            color_box = (0, 255, 0) if in_zone else (180, 180, 180)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_box, 2)
            label = names.get(clses[i], str(clses[i])) if len(clses) > i else "obj"
            conf  = confs[i] if len(confs) > i else 0.0
            idtxt = f"ID:{ids[i]}" if ids is not None and len(xyxy) == len(ids) else ""
            extra = f" | {pixel_into_meters(pix):.2f} m2" if pix is not None else ""
            put_box_label(annotated, x1, y1,
                          f"{label} {conf:.2f} {idtxt}{extra}",
                          fg=(255,255,255),
                          bg=(0,255,0) if in_zone else (70,70,70))
            cv2.circle(annotated, (cx, cy), 4, (0,0,255) if in_zone else (120,120,120), -1)

        cv2.imshow("Centroid-in-Polygon + Pixel Count", annotated)
        out.write(annotated)

        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Done. Saved: {OUT_MP4}")

if __name__ == "__main__":
    main()
