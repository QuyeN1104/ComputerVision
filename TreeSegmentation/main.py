from collections import defaultdict  # bạn đang dùng defaultdict mà chưa import
import cv2
from ultralytics import YOLO

VIDEO = "test.mp4"
MODEL = "best.pt"  # hoặc best.pt của bạn nhưng phải là model SEG
OUT   = "tracked.mp4"

def main():
    cap = cv2.VideoCapture(VIDEO)
    assert cap.isOpened(), f"Không mở được video: {VIDEO}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # tránh fps=0
    model = YOLO(MODEL)

    # Cảnh báo nếu model không phải segmentation
    if getattr(model, "task", None) != "segment":
        print("[CẢNH BÁO] Model hiện không phải 'segment'; results.masks sẽ là None.")

    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        # Lấy frame đã annotate (có mask nếu model là SEG)
        if results:
            r = results[0]
            annotated = r.plot()  # <-- vẽ masks/boxes/labels lên ảnh
            # Debug nhanh xem có mask không:
            # print("has masks:", r.masks is not None)
        else:
            annotated = frame

        if writer is None:
            h, w = annotated.shape[:2]
            writer = cv2.VideoWriter(OUT, fourcc, fps // 4, (w, h))
            assert writer.isOpened(), f"Không khởi tạo được VideoWriter: {OUT}"
            print(f"[INIT WRITER] src={w}x{h}, fps={fps}")

        writer.write(annotated)

        # Preview khung đã vẽ
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
        display = cv2.resize(annotated, (1080, 720))
        cv2.imshow("preview", display)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
