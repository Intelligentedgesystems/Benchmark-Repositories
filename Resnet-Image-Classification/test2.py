import cv2
import numpy as np
import tensorflow.lite as tflite

def load_labels(label_path):
    with open(label_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def preprocess_frame(frame):
    # Resize the frame to 299x299 and normalize it
    resized_frame = cv2.resize(frame, (299, 299))
    normalized_frame = resized_frame / 255.0  # Normalize to [0, 1]
    return np.expand_dims(normalized_frame.astype(np.float32), axis=0)

def main():
    model_path = "Inception_resnet_v2.tflite"
    label_path = "labels1.txt"

    # Load the TFLite model and allocate tensors
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load labels
    labels = load_labels(label_path)

    # Start webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Preprocess the frame
        input_data = preprocess_frame(frame)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Get the index of the highest confidence score
        predicted_index = np.argmax(output_data)
        predicted_label = labels[predicted_index]
        confidence = output_data[0][predicted_index]

        # Display the label on the frame
        cv2.putText(frame, f"{predicted_label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Webcam - Image Classification", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()