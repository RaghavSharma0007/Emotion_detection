
from flask import Flask, flash, request, redirect, url_for, render_template, Response
import cv2
from flask import Flask, flash, request, redirect, url_for, render_template, Response, send_from_directory
from flask_cors import CORS, cross_origin
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces, draw_text, draw_bounding_box, apply_offsets, load_detection_model
from utils.preprocessor import preprocess_input
import os
from io import BytesIO
from PIL import Image
import base64
from PIL import UnidentifiedImageError

app = Flask(__name__)
CORS(app)
app.secret_key = "Emotion detector"

# Parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# Hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# Loading models
face_cascade = load_detection_model('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# Getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# Starting lists for calculating modes
emotion_window = []

@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')

def PIL_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def base64_to_PIL_image(base64_img):
    return Image.open(BytesIO(base64.b64decode(base64_img)))


def detect_and_draw(image):
    im = np.array(image)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    emotion_text = None  # Initialize emotion text

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

        color = (0, 255, 0)  # Default to green for neutral/unknown emotions
        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))

        color = np.array(color).astype(int)  # Convert color to a NumPy array
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1.8, 1)  # Increase font scale from 1 to 1.8

        #draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)

    return Image.fromarray(rgb_image), emotion_text  # Return both processed image and emotion text




@app.route('/upload', methods=['GET','POST'])
@cross_origin()
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            #try:
            image = Image.open(file)
            result_image, emotion_text = detect_and_draw(image)  # Modified to also return emotion text
            result_image_base64 = PIL_image_to_base64(result_image)
            return {"image_data": result_image_base64}
                # return render_template('result.html', image_data=result_image_base64, emotion_text=emotion_text)  # Pass emotion text to template
    #         except UnidentifiedImageError:
    #             flash('Invalid image file')
    #             return redirect(request.url)
    # return render_template('upload.html')
##################################

def process_video(video_path): 
    cap = cv2.VideoCapture(video_path)
    output_path = os.path.splitext(video_path)[0] + '_output.mp4'  # Use correct file extension for output

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result_image, _ = detect_and_draw(image)  # Get processed image and ignore emotion text
            # Convert processed image to BGR and write to video
            result_frame = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            out.write(result_frame)
        else:
            break

    cap.release()
    out.release()
##########################


from flask import send_file

@app.route('/upload_video', methods=['GET','POST'])
@cross_origin()
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            video_path = os.path.join('uploads', file.filename)
            file.save(video_path)
            process_video(video_path)
            return render_template('play_video.html', video_filename=file.filename)  # Render a template with the video filename
    return render_template('upload_video.html')



@app.route('/play_video/<filename>')
def play_video(filename):
    return send_file(os.path.join('uploads', filename), mimetype='video/mp4')  # Serve the processed video file



@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', threaded=True, port=5018, debug=True)








