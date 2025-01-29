import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import cv2

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="mobilenet_v1_1.0_224_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the labels.
label_names = [line.rstrip('\n') for line in open("labels_mobilenet_quant_v1_224.txt")]

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res_im = im.resize((224, 224))
    np_res_im = np.array(res_im).astype('uint8')

    if len(np_res_im.shape) == 3:
        np_res_im = np.expand_dims(np_res_im, 0)

    # Set the tensor to the input data.
    interpreter.set_tensor(input_details[0]['index'], np_res_im)

    # Run the model.
    interpreter.invoke()

    # Get the output tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Process the output data.
    classification_prob = []
    classification_label = []
    total = 0
    for index, prob in enumerate(output_data[0]):
        if prob != 0:
            classification_prob.append(prob)
            total += prob
            classification_label.append(index)

    # Get the labels for the classified indices.
    found_labels = np.array(label_names)[classification_label]

    # Create a DataFrame with the classification results.
    df = pd.DataFrame(classification_prob / total, found_labels, columns=['Probability'])
    sorted_df = df.sort_values(by='Probability', ascending=False)

    # Print the sorted DataFrame.
    print(sorted_df)

    # Get the label with the highest probability.
    if not sorted_df.empty:
        max_label = sorted_df.index[0]
        max_prob = sorted_df.iloc[0]['Probability']
        text = f"{max_label}: {max_prob:.2f}"
        # Display the label on the frame
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
