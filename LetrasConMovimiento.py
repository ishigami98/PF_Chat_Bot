import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe import ImageFormat
import numpy as np
import matplotlib.pyplot as plt
import csv
from dtw import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

#file = open('D:\\UPAO\\SistemasGestuales&Conversacionales\\Semana15\\U.txt', 'w')


x_indice = []
y_indice = []
x_menique = []
y_menique = []
n = []
n_i = 0
#indice
with open('C:\\Users\\fze21\\OneDrive\\Documentos\\Python\\Ñ_indice.txt') as file_1:
    reader_1 = csv.reader(file_1, delimiter=' ')
    reference_1 = list(reader_1)
with open('C:\\Users\\fze21\\OneDrive\\Documentos\\Python\\S_indice.txt') as file_2:
    reader_2 = csv.reader(file_2, delimiter=' ')
    reference_2 = list(reader_2)
with open('C:\\Users\\fze21\\OneDrive\\Documentos\\Python\\Z_indice.txt') as file_3:
    reader_3 = csv.reader(file_3, delimiter=' ')
    reference_3 = list(reader_3)
with open('C:\\Users\\fze21\\OneDrive\\Documentos\\Python\\J_indice.txt') as file_4:
    reader_4 = csv.reader(file_4, delimiter=' ')
    reference_4 = list(reader_4)
#meñique

with open('C:\\Users\\fze21\\OneDrive\\Documentos\\Python\\Ñ_meñique.txt') as file_5:
    reader_5 = csv.reader(file_5, delimiter=' ')
    reference_5 = list(reader_5)
with open('C:\\Users\\fze21\\OneDrive\\Documentos\\Python\\S_meñique.txt') as file_6:
    reader_6 = csv.reader(file_6, delimiter=' ')
    reference_6 = list(reader_6)
with open('C:\\Users\\fze21\\OneDrive\\Documentos\\Python\\Z_meñique.txt') as file_7:
    reader_7 = csv.reader(file_7, delimiter=' ')
    reference_7 = list(reader_7)
with open('C:\\Users\\fze21\\OneDrive\\Documentos\\Python\\J_meñique.txt') as file_8:
    reader_8 = csv.reader(file_8, delimiter=' ')
    reference_8 = list(reader_8)


def draw_landmarks_on_image_indice(rgb_image, detection_result):
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


def draw_landmarks_on_image_menique(rgb_image, detection_result):
    global x_menique, y_menique, n, n_i
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image_m = np.copy(rgb_image)

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
            annotated_image_m,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image_m.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]

        x_menique.append(-x_coordinates[20])
        y_menique.append(-y_coordinates[20])
        n.append(n_i)
        n_i = n_i + 1
        text_x = int(min(x_coordinates) * width)
        text_y = int(max(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image_m, str(hand_landmarks[0].x) + ',' + str(hand_landmarks[0].y),
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image_m


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
    annotated_image = draw_landmarks_on_image_indice(rgb_frame.numpy_view(), detection_result)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    annotated_image_m = draw_landmarks_on_image_menique(rgb_frame.numpy_view(), detection_result)
    annotated_image_m = cv2.cvtColor(annotated_image_m, cv2.COLOR_RGB2BGR)
    #cv2.imshow('MediaPipe Hands', cv2.flip(annotated_image, 1))
    cv2.imshow('MediaPipe Hands', annotated_image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
    if cv2.waitKey(17) & 0xFF == 27:
        break
cap.release()

query_i = [x_indice, y_indice]
query_i = np.array(query_i)
query_i = query_i.T

reference_1 = np.array(reference_1)
reference_1 = reference_1.astype(float)

reference_2 = np.array(reference_2)
reference_2 = reference_2.astype(float)

reference_3 = np.array(reference_3)
reference_3 = reference_3.astype(float)

reference_4 = np.array(reference_4)
reference_4 = reference_4.astype(float)


query_m = [x_menique, y_menique]
query_m = np.array(query_m)
query_m = query_m.T

reference_5 = np.array(reference_5)
reference_5 = reference_5.astype(float)

reference_6 = np.array(reference_6)
reference_6 = reference_6.astype(float)

reference_7 = np.array(reference_7)
reference_7 = reference_7.astype(float)

reference_8 = np.array(reference_8)
reference_8 = reference_8.astype(float)

#dx = dtw(query[:,0],reference[:,0],keep_internals=True,step_pattern=rabinerJuangStepPattern(6, "c"))
#dx.plot(type='twoway')

#dy = dtw(query[:,1],reference[:,1],keep_internals=True,step_pattern=rabinerJuangStepPattern(6, "c"))
#dy.plot(type='twoway')

indice_dtw_1 = dtw(query_i, reference_1, keep_internals=True)
menique_dtw_1 = dtw(query_m, reference_5, keep_internals=True)
print(indice_dtw_1.distance)
print(menique_dtw_1.distance)

if (indice_dtw_1.distance < 15 and menique_dtw_1.distance < 15):
    print('Hiciste una Ñ')
else:
    print('No es Ñ')

indice_dtw_2 = dtw(query_i, reference_2, keep_internals=True)
menique_dtw_2 = dtw(query_m, reference_6, keep_internals=True)
print(indice_dtw_2.distance)
print(menique_dtw_2.distance)
if (indice_dtw_2.distance < 15 and menique_dtw_2.distance < 15):
    print('Hiciste una S')
else:
    print('No es S')

indice_dtw_3 = dtw(query_i, reference_3, keep_internals=True)
menique_dtw_3 = dtw(query_m, reference_7, keep_internals=True)
print(indice_dtw_3.distance)
print(menique_dtw_3.distance)
if (indice_dtw_3.distance < 15 and menique_dtw_3.distance < 15):
    print('Hiciste una Z')
else:
    print('No es Z')

indice_dtw_4 = dtw(query_i, reference_4, keep_internals=True)
menique_dtw_4 = dtw(query_m, reference_8, keep_internals=True)
print(indice_dtw_4.distance)
print(menique_dtw_4.distance)
if (indice_dtw_4.distance < 15 and menique_dtw_4.distance < 15):
    print('Hiciste una J')
else:
    print('No es J')
