import argparse
import os
import csv
from ultralytics import YOLO
import cv2

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def run_detect(source, weights, out_csv, crops_dir, conf=0.25, iou=0.45, imgsz=640):
    ensure_dir(os.path.dirname(out_csv) or '.')
    ensure_dir(crops_dir)

    model = YOLO(weights)

    results = model.predict(source=source, conf= conf, iou= iou, imgsz= imgsz, stream=True, verbose=False)

    frame_idx = -1
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'x1', 'y1', 'x2', 'y2', 'score', 'crop_path', 'img_path'])
        for r in results:
            frame_idx += 1
            img = r.orig_img
            img_path = getattr(r, 'path', '')

            if r.boxes is None or len(r.boxes) < 1:
                continue

            for b in r.boxes:
                print('DEBUG')
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                score = float(b.conf[0])

                #Crop và lưu
                h, w_img = img.shape[:2]
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(w_img, x2), min(h, y2)
                crop = img[y1c:y2c, x1c:x2c]
                crop_name = f'Frame_{frame_idx}_{x1c}_{y1c}_{x2c}_{y2c}.jpg'
                crop_path = os.path.join(crops_dir, crop_name)
                cv2.imwrite(crop_path, crop)

                w.writerow([frame_idx, x1c, y1c, x2c, y2c, score, crop_path, img_path])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument( "--source", required=True, help="Ảnh/thư mục/video/webcam (0)")
    ap.add_argument('--weights', default='C:\\Users\\USER\\OneDrive - VNU-HCMUS\\2025-3\\Automatic-License-Plate-Recognition-using-YOLOv8\\license_plate_detector.pt')
    ap.add_argument('--out_csv', default='detections.csv')
    ap.add_argument('--crops_dir', default='crops')
    ap.add_argument('--conf', type=float,default=0.25)
    ap.add_argument('--iou', type=float,default=0.45)
    ap.add_argument('--imgsz', type=int, default=640)
    args = ap.parse_args()

    run_detect(**vars(args))


