import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="lite-model_deeplabv3_1_metadata_2.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

# Define the labels
labelsArrays = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv",
]

def process_frame(frame):
    # Convert BGR to RGB (cv2 uses BGR by default)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image and resize
    pil_image = Image.fromarray(rgb_frame)
    res_im = pil_image.resize((257, 257))

    # Convert to numpy array and normalize
    np_res_im = np.array(res_im)
    np_res_im = (np_res_im / 255).astype("float32")
    np_res_im = np.expand_dims(np_res_im, 0)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]["index"], np_res_im)

    # Run the model
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]["index"])

    # Process the output data
    mSegmentBits = np.zeros((257, 257)).astype(int)
    outputbitmap = np.zeros((257, 257)).astype(int)

    for y in range(257):
        for x in range(257):
            maxVal = 0
            mSegmentBits[x][y] = 0
            for c in range(21):
                value = output_data[0][y][x][c]
                if c == 0 or value > maxVal:
                    maxVal = value
                    mSegmentBits[y][x] = c
            if mSegmentBits[y][x] == 15:  # 15 is the index for "person"
                outputbitmap[y][x] = 1
            else:
                outputbitmap[y][x] = 0

    # Convert the output bitmap to an image
    temp_outputbitmap = outputbitmap * 255
    mask = np.uint8(temp_outputbitmap)

    # Resize mask to match original frame size
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    return mask


try:
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Process the frame
        mask = process_frame(frame)

        # Create a color mask for visualization
        color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Overlay the mask on the original frame with reduced opacity
        blended = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)

        # Display the blended output in a single window
        cv2.imshow("Blended Output", blended)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
