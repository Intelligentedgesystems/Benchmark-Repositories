import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import time
import cv2

model_path = 'yolov8m_float32.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Obtain the height and width of the corresponding image from the input tensor
image_height = input_details[0]['shape'][1] # 640
image_width = input_details[0]['shape'][2] # 640

# Threshold Setting
threshold = 0.3

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Image Preparation
    image_resized = cv2.resize(frame, (image_width, image_height))
    image_np = np.array(image_resized)
    image_np = np.true_divide(image_np, 255, dtype=np.float32)
    image_np = image_np[np.newaxis, :]

    # Inference
    interpreter.set_tensor(input_details[0]['index'], image_np)

    start = time.time()
    interpreter.invoke()
    print(f'run time：{time.time() - start:.2f}s')

    # Obtaining output results
    output = interpreter.get_tensor(output_details[0]['index'])
    output = output[0]
    output = output.T

    boxes_xywh = output[..., :4] # Get coordinates of bounding box, first 4 columns of output tensor
    scores = np.max(output[..., 5:], axis=1) # Get score value, 5th column of output tensor
    classes = np.argmax(output[..., 5:], axis=1) # Get the class value, get the 6th and subsequent columns of the output tensor, and store the largest value in the output tensor.

    # Draw bounding boxes, scores, and classes on the image
    for box, score, cls in zip(boxes_xywh, scores, classes):
        if score >= threshold:
            x_center, y_center, width, height = box
            x1 = int((x_center - width / 2) * image_width)
            y1 = int((y_center - height / 2) * image_height)
            x2 = int((x_center + width / 2) * image_width)
            y2 = int((y_center + height / 2) * image_height)

            cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = f"Class: {cls}, Score: {score:.2f}"
            cv2.putText(image_resized, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the image
    cv2.imshow('Object Detection', image_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()