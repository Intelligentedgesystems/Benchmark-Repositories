import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="ssd_mobilenet_v2_coco.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Input and output tensor details
input_index = input_details[0]['index']  # Input tensor index (Preprocessor/sub)
output_boxes_index = output_details[0]['index']  # Output tensor index (concat: bounding boxes)
output_classes_index = output_details[1]['index']  # Output tensor index (concat_1: class scores)

# Load label names
label_names = [line.rstrip('\n') for line in open("coco_labels_list.txt")]

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res_im = im.resize((300, 300))
    np_res_im = np.array(res_im).astype('float32')
    np_res_im = np.expand_dims(np_res_im, axis=0)  # Add batch dimension

    # Normalize input
    np_res_im = np_res_im / 255.0

    # Perform inference
    interpreter.set_tensor(input_index, np_res_im)
    interpreter.invoke()

    # Retrieve output tensors
    output_boxes = interpreter.get_tensor(output_boxes_index)  # Shape: [1, 1917, 1, 4]
    output_classes = interpreter.get_tensor(output_classes_index)  # Shape: [1, 1917, 91]

    height, width, _ = frame.shape

    # Iterate through detections
    for i in range(output_boxes.shape[1]):  # Iterate over 1917 detections
        bbox = output_boxes[0, i, 0]  # Extract bounding box coordinates

        # Skip invalid bounding boxes
        if len(bbox) != 4 or not np.all(np.isfinite(bbox)):
            continue

        ymin, xmin, ymax, xmax = bbox
        class_scores = output_classes[0, i]
        max_class_index = np.argmax(class_scores)
        max_class_score = class_scores[max_class_index]

        # Only consider detections with a high confidence score
        if max_class_score > 0.5:
            # Improve bounding box visualization by adding a margin to the boxes
            margin = 5  # Add a small margin to bounding box dimensions
            xmin = max(0, int(xmin * width) - margin)
            xmax = min(width, int(xmax * width) + margin)
            ymin = max(0, int(ymin * height) - margin)
            ymax = min(height, int(ymax * height) + margin)

            # Get the class label
            label = label_names[max_class_index] if max_class_index < len(label_names) else "Unknown"

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {max_class_score:.2f}", (xmin, max(0, ymin - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with proper color rendering
    cv2.imshow('Webcam', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
