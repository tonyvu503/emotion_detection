#!/usr/bin/env python
# coding: utf-8

# In[3]:


# !pip install streamlit


# In[1]:
pip install onnx
pip install onnxruntime

import onnx
import onnxruntime as rt
import numpy as np
import cv2
import os
import time


# In[4]:


import streamlit as st


# In[5]:


@st.cache()
def load_model():
    model = rt.InferenceSession("FER_Model_Adam.onnx")
    return model


# In[6]:


@st.cache()
def classes():
    classes = ['fear', 'angry', 'sad', 'neutral', 'surprise', 'disgust', 'happy']
    return classes


# In[7]:


@st.cache()
def emotion_detection():
    # Face detection setup
    cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(cascPathface)
    video_capture = cv2.VideoCapture(0)
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


#             start = time.time()
            pred_onx = model.run([output_name], {input_name: face_input})
#             end = time.time()
#             print("Time taken by onnx model: ", end - start)
            print("Emotion: ", classes[np.argmax(pred_onx)])

            cv2.putText(frame, classes[np.argmax(pred_onx)], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    return None


# In[ ]:


if __name__ == '__main__':
    model = load_model()
    classes = classes()
    st.title('Welcome To Project Emotion Detection')
    guidance = """Click on the Start button to start the live emotion detection. Press Q to escape"""
    st.write(guidance)
    button = st.button("Start Emotion Detection")
    if button:
        emotion_detection()

