import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="detr_resnet50_dc5.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Extract input/output shapes and types
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
output_logits_idx = next(idx for idx, d in enumerate(output_details) if d['name'] == 'logits')
output_boxes_idx = next(idx for idx, d in enumerate(output_details) if d['name'] == 'boxes')

# Helper function to preprocess frames
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (480, 480))
    input_data = np.expand_dims(resized_frame, axis=0).astype(input_dtype)
    input_data = input_data / 255.0  # Normalize to [0, 1]
    return input_data

# Helper function to draw boxes and labels
def draw_boxes(frame, boxes):
    for box in boxes:
        ymin, xmin, ymax, xmax = box
        xmin = int(xmin * frame.shape[1])
        xmax = int(xmax * frame.shape[1])
        ymin = int(ymin * frame.shape[0])
        ymax = int(ymax * frame.shape[0])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

# Open webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

    # Preprocess the frame
    input_data = preprocess_frame(frame)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get outputs
    logits = interpreter.get_tensor(output_details[output_logits_idx]['index'])
    boxes = interpreter.get_tensor(output_details[output_boxes_idx]['index'])

    # Draw bounding boxes
    draw_boxes(frame, boxes[0])

    # Display the frame
    cv2.imshow("Webcam Feed", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
