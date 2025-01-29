import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="mobilenet_v1_1.0_224_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the image.
im = Image.open("chihuahua.jpg")
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

# Load the labels.
label_names = [line.rstrip('\n') for line in open("labels_mobilenet_quant_v1_224.txt")]

# Get the labels for the classified indices.
found_labels = np.array(label_names)[classification_label]

# Create a DataFrame with the classification results.
df = pd.DataFrame(classification_prob / total, found_labels, columns=['Probability'])
sorted_df = df.sort_values(by='Probability', ascending=False)

# Print the sorted DataFrame.
print(sorted_df)
