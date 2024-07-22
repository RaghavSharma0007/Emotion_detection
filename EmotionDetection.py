from flask import Flask, flash, request, redirect, url_for, render_template,Response
import cv2
from flask_cors import CORS,cross_origin
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import pandas as pd
import sys
import os
import re
from io import BytesIO
from PIL import Image
import base64
from PIL import UnidentifiedImageError

USE_WEBCAM = True # If false, loads video file source

app = Flask(__name__)
CORS(app)
app.secret_key = "Emotion detector"

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming

#cv2.namedWindow('window_frame')
#video_capture = cv2.VideoCapture(0)

# Select video or webcam feed
#cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0) # Webcam source
else:
    cap = cv2.VideoCapture('./demo/dinner.mp4') # Video file source
@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')

# #####################################################################################################
def PIL_image_to_base64(pil_image):
    buffered = BytesIO()
    img1=pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue())


def base64_to_PIL_image(base64_img):
    return Image.open(BytesIO(base64.b64decode(base64_img)))
@cross_origin()
def detect_and_draw(image_string):
    try :
        image = base64_to_PIL_image(image_string)
    except UnidentifiedImageError:
        return process()
    im = np.array(image)
    # input_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.flip(im, 1)
    #####################################################
    if im is not None:
        gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for face_coordinates in faces:

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue


            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,color, 0, -45, 1, 1)

        # img2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        print("Emotion: " + str(emotion_text) + " emotion_probability " + str(emotion_probability))
        img=rgb_image
        # img = cv2.flip(img, 1)
        # imageRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # imageRGB = input_img
        imageRGB = img[..., ::-1]
        # cv2.imwrite("test2.png", imageRGB)
        PIL_image = Image.fromarray(imageRGB)
    return PIL_image_to_base64(PIL_image)
###################################WEBCAM###################################################
@app.route('/webcam')
@cross_origin()
def index():
    return render_template('layout.html')


@app.route('/process', methods=['POST'])
@cross_origin()
def process():
    input = request.json
    image_data = re.sub('^data:image/.+;base64,', '', input['img'])
    image_ascii = detect_and_draw(image_data)
    return image_ascii

###################################WEBCAM-END###################################################

if __name__ == "__main__":
    app.run(host='0.0.0.0',threaded=True, port=5018,debug=True)