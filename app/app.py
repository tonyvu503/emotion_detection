#!/usr/bin/env python
# coding: utf-8

# In[3]:


# !pip install streamlit


# In[1]:

from types import NoneType
from flask import Flask, render_template, request
import onnx
import onnxruntime as rt
import numpy as np
import cv2
import os
import time
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='./templates')
@app.route("/")
def index():
    return render_template("index.html")

def update_emotion_bars(sleep_time, emotion_probs, frame, x, y, w, last_update_time, classes):
    global last_emotion_prob
    current_time = time.time()
    last_emotion_prob = None
    if current_time - last_update_time > sleep_time:
        # Update the emotion_probs with the latest values
        emotion_probs = emotion_probs
        last_emotion_prob = emotion_probs
        last_update_time = current_time
    else:
        # Keep the emotion_probs as it is
        if(last_emotion_prob is not None):
            emotion_probs = last_emotion_prob

    # Set up graph parameters
    bar_width = 20
    bar_spacing = 10
    max_bar_height = 100
    x_offset = x + w + 10
    y_offset = y

    # Draw the probability bars
    for i, prob in enumerate(emotion_probs):
        bar_height = int(prob * max_bar_height)
        bar_color = (0, 255, 0) if i == np.argmax(emotion_probs) else (0, 0, 255)
        cv2.rectangle(frame, (x_offset, y_offset + i * (bar_width + bar_spacing)), (x_offset + bar_height, y_offset + (i + 1) * bar_width + i * bar_spacing), bar_color, -1)
        cv2.putText(frame, f"{classes[i]}: {prob:.2f}", (x_offset + bar_height + 5, y_offset + (i + 1) * bar_width + i * bar_spacing - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame, last_update_time

@app.route("/emotion_detection", methods=["POST"])
def emotion_detection():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, "FER_Model_Adam.onnx")
    model = rt.InferenceSession(model_path)
    # Face detection setup
    cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(cascPathface)
    video_capture = cv2.VideoCapture(0)
    last_update_time = time.time()
    last_emotion_prob = None
    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            face_gray = gray[y:y+h, x:x+w]
            face_gray_resized = cv2.resize(face_gray, (128, 128))
            
            # Convert the single channel grayscale image to a 3-channel image
            face_gray_3channel = cv2.cvtColor(face_gray_resized, cv2.COLOR_GRAY2BGR)

            # Prepare the input for the model
            face_input = np.transpose(face_gray_3channel, (2, 0, 1))
            face_input = np.expand_dims(face_input, axis=0)
            face_input = face_input.astype(np.float32)
            face_input = face_input / 255.0


            classes = ['fear', 'angry', 'sad', 'neutral', 'surprise', 'disgust', 'happy']
            input_name = model.get_inputs()[0].name
            output_name = model.get_outputs()[0].name
            pred_onx = model.run([output_name], {input_name: face_input})

            print("Emotion: ", classes[np.argmax(pred_onx)])

            cv2.putText(frame, classes[np.argmax(pred_onx)], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            # Get the probabilities for each emotion class
            emotion_probs = pred_onx[0][0]

            # Call the update_emotion_bars function with the sleep time and emotion_probs
            frame, last_update_time = update_emotion_bars(0.5, emotion_probs, frame, x, y, w, last_update_time,classes)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    return render_template('index.html')

# In[ ]:


if __name__ == '__main__':
    app.run(port=8000)


