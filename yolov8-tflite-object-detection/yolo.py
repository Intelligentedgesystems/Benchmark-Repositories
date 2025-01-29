from ultralytics import YOLO

model = YOLO('yolov8m.pt')
results = model.export(format='tflite', imgsz=320)  # export to TensorFlow Lite