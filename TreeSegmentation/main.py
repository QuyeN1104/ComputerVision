import cv2
from ultralytics import YOLO
from collections import defaultdict

def YOLO_largest_seg(input_path:str, output:str, model:str='best.pt'):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = None
    model = YOLO(model)

    max_area_by_id = defaultdict(int)
    best_global = {'area': 0, 'trackid': None, 'frame': None}
    window = cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            print('Not Oke')
            return

        results = model.track(frame, verbose=False, persist=True, conf=0.5)

        if not results:
            annotated = frame
        else:
            r = results[0]
            annotated = r.plot()
            h, w = annotated.shape[:2]


            # Calculate area
            if r.masks is not None:
                masks = r.masks.data
                ids = r.boxes.id
                # print(f'[DEBUG] {masks}')
                # print(f'[DEBUG] {ids}')
                for i, mask in enumerate(masks):
                    area_px = int((mask > 0.5).sum())

                    if ids is not None:
                        tid = int(ids[i])
                        if area_px > max_area_by_id[tid]:
                            max_area_by_id[tid] = area_px

                            if area_px > best_global['area']:
                                best_global.update({'area': area_px, 'trackid': tid, 'frame': (annotated.copy(), frame.copy())})

                text = f"Max Area = {best_global['area']}, track_id = {best_global['trackid']}"
                cv2.putText(
                    annotated,
                    text,
                    (10, 30),  # cách trái 10px, cách trên ~30px
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,  # font scale
                    (0, 255, 255),  # màu (BGR) → vàng
                    2,  # độ dày nét
                    cv2.LINE_AA
                )

        # Write Video
        if writer is None:
            writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        writer.write(annotated)

        cv2.imshow('frame', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if best_global['area'] > 0:
        print(f'The largest area is {best_global["area"]} pixels, trackid is {best_global["trackid"]}')
        # mkdir('results')
        cv2.imwrite('annotated.jpg', best_global['frame'][0])
        cv2.imwrite('original.jpg', best_global['frame'][1])
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

YOLO_largest_seg('test.mp4', 'output.mp4')