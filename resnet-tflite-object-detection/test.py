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
input_shape = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]
output_logits_idx = next(
    idx for idx, d in enumerate(output_details) if d["name"] == "logits"
)
output_boxes_idx = next(
    idx for idx, d in enumerate(output_details) if d["name"] == "boxes"
)

# Configuration parameters
CONFIDENCE_THRESHOLD = 0.7  # Adjust this value based on your needs
BACKGROUND_CLASS_ID = 91  # Adjust according to your model's class setup


# Helper function to preprocess frames
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (480, 480))
    input_data = np.expand_dims(resized_frame, axis=0).astype(input_dtype)
    input_data = input_data / 255.0  # Normalize to [0, 1]
    return input_data


# Modified helper function to filter and draw boxes
def draw_filtered_boxes(frame, logits, boxes):
    # Process logits to get class probabilities
    scores = tf.nn.softmax(logits[0], axis=-1).numpy()
    max_scores = np.max(scores, axis=1)
    max_classes = np.argmax(scores, axis=1)

    filtered_boxes = []
    for i in range(len(max_scores)):
        # Filter based on confidence and exclude background class
        if (
            max_scores[i] > CONFIDENCE_THRESHOLD
            and max_classes[i] != BACKGROUND_CLASS_ID
        ):
            box = boxes[0][i]
            ymin, xmin, ymax, xmax = box

            # Convert normalized coordinates to frame dimensions
            xmin = int(xmin * frame.shape[1])
            xmax = int(xmax * frame.shape[1])
            ymin = int(ymin * frame.shape[0])
            ymax = int(ymax * frame.shape[0])

            # Draw rectangle and label
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            # Optional: Add class label text
            # cv2.putText(frame, f"Class {max_classes[i]}", (xmin, ymin-10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)


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
    interpreter.set_tensor(input_details[0]["index"], input_data)

    # Run inference
    interpreter.invoke()

    # Get outputs
    logits = interpreter.get_tensor(output_details[output_logits_idx]["index"])
    boxes = interpreter.get_tensor(output_details[output_boxes_idx]["index"])

    # Draw filtered bounding boxes
    draw_filtered_boxes(frame, logits, boxes)

    # Display the frame
    cv2.imshow("Webcam Feed", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
