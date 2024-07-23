import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe import ImageFormat
import numpy as np
import matplotlib.pyplot as plt
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

file = open('C:\\Users\\fze21\\OneDrive\\Documentos\\Python\\S_indice.txt', 'w')

x_indice = []
y_indice = []
n = []
n_i = 0

def draw_landmarks_on_image(rgb_image, detection_result):
    global x_indice, y_indice, n, n_i
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]

        x_indice.append(-x_coordinates[8])
        y_indice.append(-y_coordinates[8])
        n.append(n_i)
        n_i = n_i + 1
        text_x = int(min(x_coordinates) * width)
        text_y = int(max(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, str(hand_landmarks[0].x) + ',' + str(hand_landmarks[0].y),
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='C:\\Users\\fze21\\OneDrive\\Documentos\\Python\\hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# For webcam input:
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = image[:, :, ::-1]
    #print(image.shape)
    rgb_frame = mp.Image(image_format=ImageFormat.SRGB, data=image)

    # Draw the hand annotations on the image.
    #image.flags.writeable = True
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #image_height, image_width, _ = image.shape
    # STEP 4: Detect hand landmarks from the input image.
    detection_result = detector.detect(rgb_frame)

    # STEP 5: Process the classification result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(rgb_frame.numpy_view(), detection_result)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    #cv2.imshow('MediaPipe Hands', cv2.flip(annotated_image, 1))
    cv2.imshow('MediaPipe Hands', annotated_image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()

for i in range(len(x_indice)):
    #print(n[i],':',x_index[i],',',y_index[i])
    #print(x_indice[i],' ',y_indice[i])
    file.write(str(x_indice[i]) + ' ' + str(y_indice[i]) + '\n')

fig, ax = plt.subplots()
ax.scatter(x_indice, y_indice)

for i in range(len(x_indice)):
    ax.annotate(str(n[i]), (x_indice[i], y_indice[i]))

plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.show()

file.close()
