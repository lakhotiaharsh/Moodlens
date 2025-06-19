from statistics import mode
import cv2
from keras.models import load_model
import numpy as np

from dataset import get_labels
from grad_cam import detect_faces
from grad_cam import draw_text
from grad_cam import draw_bounding_box
from grad_cam import apply_offsets
from grad_cam import load_detection_model
from preprocessing import preprocess_input

# parameters for loading data and images
detection_model_path = 'trained_models/haarcascade_frontalface_default.xml'
emotion_model_path = 'trained_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = 'trained_models/simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
frame_window = 10
gender_offsets = (30, 60)
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

# starting lists for calculating modes
gender_window = []
emotion_window = []

# starting video streaming
video_path = 'input.mp4'  # Replace with the path to your video file
video_capture = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
output_path = 'output_video.mp4'  # Replace with your desired output path and filename
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # You can change the codec if needed (e.g., 'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:

    ret, bgr_image = video_capture.read() # Read frame and check if successful
    if not ret: # If frame not read, break the loop
        break
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            rgb_face = cv2.resize(rgb_face, (gender_target_size))
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        # Convert rgb_face to grayscale before processing for gender classification
        gray_face_for_gender = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2GRAY)
        gray_face_for_gender = preprocess_input(gray_face_for_gender, False)
        gray_face_for_gender = np.expand_dims(gray_face_for_gender, 0)
        gray_face_for_gender = np.expand_dims(gray_face_for_gender, -1)
        gender_prediction = gender_classifier.predict(gray_face_for_gender)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]
        gender_window.append(gender_text)


        gray_face = preprocess_input(gray_face, False)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)


        if len(gender_window) > frame_window:
            emotion_window.pop(0)
            gender_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
            gender_mode = mode(gender_window)
        except:
            continue

        if gender_text == gender_labels[0]:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, gender_mode,
                  color, 0, -20, 1, 1)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    out.write(bgr_image) # Write the processed frame to the output video file

video_capture.release()
out.release() # Release the VideoWriter