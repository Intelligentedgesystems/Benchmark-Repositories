import numpy as np
import tensorflow as tf
from PIL import Image

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="lite-model_deeplabv3_1_metadata_2.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the image
im = Image.open("image.jpg")
res_im = im.resize((257, 257))
np_res_im = np.array(res_im)
np_res_im = (np_res_im / 255).astype('float32')

if len(np_res_im.shape) == 3:
    np_res_im = np.expand_dims(np_res_im, 0)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], np_res_im)

# Run the model
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Define the labels
labelsArrays = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
                "person", "potted plant", "sheep", "sofa", "train", "tv"]

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
        label = labelsArrays[mSegmentBits[x][y]]
        if mSegmentBits[y][x] == 15:
            outputbitmap[y][x] = 1
        else:
            outputbitmap[y][x] = 0

# Convert the output bitmap to an image
temp_outputbitmap = outputbitmap * 255
PIL_image = Image.fromarray(np.uint8(temp_outputbitmap)).convert('L')

# Resize the mask to the original image size
org_mask_img = PIL_image.resize(im.size)

# Save the output image
org_mask_img.save("output_image.png")
