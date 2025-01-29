import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import math
from enum import Enum

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the image
im = Image.open("gun.jpg")
res_im = im.resize((257, 257))
np_res_im = np.array(res_im)
np_res_im = (np_res_im / 255).astype('float32')

if len(np_res_im.shape) == 3:
    np_res_im = np.expand_dims(np_res_im, 0)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], np_res_im)
interpreter.invoke()

# Get the output tensors
heatmaps = interpreter.get_tensor(output_details[0]['index'])
offsets = interpreter.get_tensor(output_details[1]['index'])
forward_displacements = interpreter.get_tensor(output_details[2]['index'])
backward_displacements = interpreter.get_tensor(output_details[3]['index'])

# Extract keypoint positions
height = heatmaps[0].shape[0]
width = heatmaps[0][0].shape[0]
numKeypoints = heatmaps[0][0][0].shape[0]

keypointPositions = []
for keypoint in range(numKeypoints):
    maxVal = -float('inf')
    maxRow = 0
    maxCol = 0
    for row in range(height):
        for col in range(width):
            if heatmaps[0][row][col][keypoint] > maxVal:
                maxVal = heatmaps[0][row][col][keypoint]
                maxRow = row
                maxCol = col
                maxRow = row
                maxCol = col
    keypointPositions.append([maxRow, maxCol])

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

confidenceScores = []
yCoords = []
xCoords = []
for idx, position in enumerate(keypointPositions):
    positionY = keypointPositions[idx][0]
    positionX = keypointPositions[idx][1]
    yCoords.append(position[0] / (height - 1) * 257 + offsets[0][positionY][positionX][idx])
    xCoords.append(position[1] / (width - 1) * 257 + offsets[0][positionY][positionX][idx + numKeypoints])
    confidenceScores.append(sigmoid(heatmaps[0][positionY][positionX][idx]))

score = np.average(confidenceScores)

class BodyPart(Enum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

bodyJoints = np.array([
    (BodyPart.LEFT_WRIST, BodyPart.LEFT_ELBOW),
    (BodyPart.LEFT_ELBOW, BodyPart.LEFT_SHOULDER),
    (BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER),
    (BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW),
    (BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST),
    (BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP),
    (BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP),
    (BodyPart.RIGHT_HIP, BodyPart.RIGHT_SHOULDER),
    (BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE),
    (BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE),
    (BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE),
    (BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE)
])

# Plot and save the result
minConfidence = 1.0
fig, ax = plt.subplots(figsize=(10, 10))

if score < minConfidence:
    ax.imshow(res_im)
    for line in bodyJoints:
        plt.plot([xCoords[line[0].value], xCoords[line[1].value]], [yCoords[line[0].value], yCoords[line[1].value]], 'k-')
    ax.scatter(xCoords, yCoords, s=30, color='r')
    plt.savefig("pose_estimation_output.png")
    plt.show()
