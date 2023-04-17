#!/usr/bin/env python
# coding: utf-8

# In[3]:


# !pip install streamlit


# In[1]:


import onnx
import onnxruntime as rt
import numpy as np
import cv2
import os
import time


# In[2]:


import streamlit as st


# In[49]:


@st.cache
def image_load(file):
    image = cv2.imread(file)
    img = image.copy()
    return img


# In[ ]:


@st.cache()
def load_model():
    model = rt.InferenceSession("FER_Model_Adam.onnx")
    return model


# In[57]:


@st.cache
def image_show(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[54]:


@st.cache()
def image_emotion_detection(img):
    # Face detection setup
    cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(cascPathface)
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        face_gray = gray[y:y+h, x:x+w]
        face_gray_resized = cv2.resize(face_gray, (128, 128))
            
        # Convert the single channel grayscale image to a 3-channel image
        face_gray_3channel = cv2.cvtColor(face_gray_resized, cv2.COLOR_GRAY2BGR)

        # Prepare the input for the model
        face_input = np.transpose(face_gray_3channel, (2, 0, 1))
        face_input = np.expand_dims(face_input, axis=0)
        face_input = face_input.astype(np.float32)
        face_input = face_input / 255.0
        #model prediction
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        classes = ['fear', 'angry', 'sad', 'neutral', 'surprise', 'disgust', 'happy']
        pred_onx = model.run([output_name], {input_name: face_input})
#         print("Emotion: ", classes[np.argmax(pred_onx)])
        cv2.putText(img, classes[np.argmax(pred_onx)], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return img


# In[ ]:


if __name__ == '__main__':
    model = load_model()
    st.title('Welcome To Project Emotion Detection')
    guidance = """upload your image and review the emotion detected"""
    st.write(guidance)
    file = st.file_uploader('Upload An Image')
    if file:
        img = image_load(file)
        img = image_emotion_detection(img)
    
    st.title('Here are the emotion detected image')
    st.image(img)

