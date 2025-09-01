from ultralytics import YOLO

model = YOLO('best (6).pt')

model.track('tomato_2.MOV', show=True)