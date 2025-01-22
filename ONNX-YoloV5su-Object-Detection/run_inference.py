# import cv2
# import numpy as np
# import onnxruntime as ort

# # Load YOLOv5 model
# def load_model(model_path):
#     return ort.InferenceSession(model_path)

# # Perform object detection
# def detect_objects(session, image, input_size=(320, 320)):
#     # Preprocess image
#     h, w = image.shape[:2]
#     resized = cv2.resize(image, input_size)
#     blob = resized.transpose(2, 0, 1).astype(np.float32) / 255.0  # Normalize
#     blob = np.expand_dims(blob, axis=0)

#     # Run inference
#     inputs = {session.get_inputs()[0].name: blob}
#     outputs = session.run(None, inputs)
#     return outputs, (w, h)

# # Post-process detections
# def post_process(outputs, img_shape, conf_threshold=0.25, iou_threshold=0.45):
#     # Parse outputs
#     predictions = outputs[0]
#     print(f"Predictions shape: {predictions.shape}")  # Debug print
#     boxes = predictions[:, :4]  # x1, y1, x2, y2
#     scores = predictions[:, 4] * predictions[:, 5:].max(axis=1)  # Class confidence
#     classes = predictions[:, 5:].argmax(axis=1)  # Class indices

#     # Filter by confidence threshold
#     indices = np.where(scores > conf_threshold)[0]
#     boxes = boxes[indices]
#     scores = scores[indices]
#     classes = classes[indices]

#     # Scale boxes back to original image size
#     w, h = img_shape
#     scale_boxes = lambda box: [int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)]
#     boxes = np.array([scale_boxes(box) for box in boxes])

#     # Non-max suppression
#     indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
#     if len(indices) > 0:
#         indices = indices.flatten()
#         boxes = boxes[indices]
#         scores = scores[indices]
#         classes = classes[indices]
#     else:
#         boxes = np.array([])
#         scores = np.array([])
#         classes = np.array([])

#     return boxes, scores, classes

# # Main function to run object detection
# def main():
#     model_path = "yolov5su.onnx"
#     session = load_model(model_path)

#     cap = cv2.VideoCapture(0)  # Open webcam
#     if not cap.isOpened():
#         print("Error: Cannot open webcam.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to read frame from webcam.")
#             break

#         # Run detection
#         outputs, img_shape = detect_objects(session, frame)
#         print(f"Outputs: {outputs}")  # Debug print
#         boxes, scores, classes = post_process(outputs, img_shape)

#         # Draw detections on the frame
#         for box, score, cls in zip(boxes, scores, classes):
#             x1, y1, x2, y2 = box
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(img_shape[1], x2), min(img_shape[0], y2)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             label = f"Class {cls}: {score:.2f}"
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Display the frame
#         cv2.imshow("YOLOv5 Object Detection", frame)

#         # Exit on pressing 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


import cv2

from yolov8 import YOLOv8

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize YOLOv7 object detector
model_path = "yolov5su.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)

    combined_img = yolov8_detector.draw_detections(frame)
    cv2.imshow("Detected Objects", combined_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
