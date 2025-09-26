import argparse
import csv
import easyocr

def normalize_plate(text):
    return text.upper()


def run_ocr(detection_csv, out_csv):
    ocr = easyocr.Reader(['en'], gpu=False)

    with open(detection_csv, newline='') as file_in, open(out_csv, mode='w') as file_out:
        reader = csv.DictReader(file_in)
        writer = csv.writer(file_out)
        writer.writerow(['frame', 'plate_text', 'conf', 'crop_path'])

        for row in reader:
            crop_path = row['crop_path']
            res = ocr.readtext(crop_path)[0]
            bbox, text, conf = res
            text = normalize_plate(text)
            writer.writerow([row['frame'], text, conf, crop_path])
        print('Successully Saved')
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", default="detections.csv")
    ap.add_argument("--out", default="reads.csv")
    args = ap.parse_args()

    run_ocr(args.detections, args.out)