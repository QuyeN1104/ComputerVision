import argparse
import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr


# =============== TIỆN ÍCH ===============

def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)


def normalize_plate(text: str) -> str:
    """Chuẩn hóa cơ bản: viết hoa, bỏ khoảng trắng, map O->0, I/L->1, Z->2, B->8, giữ A-Z0-9-."""
    if not text:
        return ""
    t = text.strip().upper().replace(" ", "")
    mapping = {"O": "0", "I": "1", "L": "1", "Z": "2", "B": "8"}
    t = "".join(mapping.get(ch, ch) for ch in t)
    # chỉ giữ A-Z, 0-9 và dấu '-'
    t = "".join(ch for ch in t if ch.isalnum() or ch == "-")
    return t


# =============== PIPELINE ===============

def process_source(
    source,
    yolo,
    ocr: easyocr.Reader,
    crops_dir: str,
    out_csv: str,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    save_video: str | None = None,
):
    """Video/ảnh/webcam -> detect -> crop -> EasyOCR -> CSV (+ overlay)"""
    ensure_dir(crops_dir)
    ensure_dir(os.path.dirname(out_csv) or ".")

    # writer video (nếu cần)
    vw = None

    # Ultralytics stream ra từng frame
    results = yolo.predict(
        source=source, conf=conf, iou=iou, imgsz=imgsz, stream=True, verbose=False
    )

    frame_idx = -1
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "plate_text", "conf", "x1", "y1", "x2", "y2", "crop_path"])

        for det_out in results:
            frame_idx += 1
            frame = det_out.orig_img

            # Khởi tạo video writer ở frame đầu tiên nếu có yêu cầu lưu
            if save_video and vw is None and frame is not None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vw = cv2.VideoWriter(save_video, fourcc, 25, (frame.shape[1], frame.shape[0]))

            if det_out.boxes is None or len(det_out.boxes) == 0:
                if vw is not None:
                    vw.write(frame)
                continue

            # Duyệt qua các bbox của YOLO
            for box in det_out.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Cắt crop an toàn trong khung
                h, w = frame.shape[:2]
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(w, x2), min(h, y2)
                if x2c <= x1c or y2c <= y1c:
                    continue

                crop = frame[y1c:y2c, x1c:x2c]
                if crop is None or crop.size == 0:
                    continue
                if crop.shape[0] < 10 or crop.shape[1] < 20:
                    # quá nhỏ, OCR sẽ kém/đổ lỗi
                    continue

                # Lưu crop (hữu ích để debug và huấn luyện OCR sau này)
                crop_name = f"{frame_idx}_{x1c}_{y1c}_{x2c}_{y2c}.jpg"
                crop_path = os.path.join(crops_dir, crop_name)
                ok = cv2.imwrite(crop_path, crop)
                if not ok:
                    # nếu không ghi được, vẫn OCR trực tiếp từ mảng
                    crop_path = ""

                # ===== EASYOCR =====
                # Chuyển sang ảnh xám 2D để tránh lỗi "too many values to unpack"
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

                # Cho phép chỉ A-Z, 0-9, '-' để giảm nhiễu
                results_ocr = ocr.readtext(
                    gray,
                    detail=1,                         # [(bbox, text, conf), ...]
                    paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
                )

                if not results_ocr:
                    # ghi lại dòng rỗng để bạn còn biết có bbox nhưng không đọc được
                    writer.writerow([frame_idx, "", 0.0, x1c, y1c, x2c, y2c, crop_path])
                    # vẽ bbox rỗng (tùy chọn)
                    cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), (0, 255, 255), 2)
                    continue

                # Với biển số, thường chỉ 1 dòng -> lấy kết quả có conf cao nhất
                best = max(results_ocr, key=lambda t: float(t[2]) if len(t) >= 3 else 0.0)
                bbox_e, text_raw, conf_e = best  # (bbox4điểm, text, conf)
                text_norm = normalize_plate(text_raw)

                # Ghi CSV
                writer.writerow([frame_idx, text_norm, float(conf_e), x1c, y1c, x2c, y2c, crop_path])

                # Vẽ overlay
                cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{text_norm} ({float(conf_e):.2f})",
                    (x1c, max(0, y1c - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            if vw is not None:
                vw.write(frame)

    if vw is not None:
        vw.release()


# =============== CLI ===============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="đường dẫn ảnh/thư mục/video hoặc webcam=0")
    ap.add_argument("--weights", default="resource/model_license.pt")
    ap.add_argument("--crops_dir", default="crops")
    ap.add_argument("--out_csv", default="reads_e2e.csv")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--save_video", default=None, help="đường dẫn .mp4 để lưu video overlay")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--gpu", action="store_true", help="dùng GPU cho EasyOCR nếu sẵn sàng")
    args = ap.parse_args()

    # Hỗ trợ nhập webcam=0
    src = 0 if str(args.source).strip() == "0" else args.source

    # Load models
    yolo = YOLO(args.weights)
    ocr = easyocr.Reader([args.lang], gpu=args.gpu)

    process_source(
        source=src,
        yolo=yolo,
        ocr=ocr,
        crops_dir=args.crops_dir,
        out_csv=args.out_csv,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save_video=args.save_video,
    )


if __name__ == "__main__":
    main()
