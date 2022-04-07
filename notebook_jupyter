# IN JUPYTER LAB

## -----app.py-----
```
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image
from yaml import load, Loader, dump


#IMAGE_WIDTH = 128
#IMAGE_HEIGHT = IMAGE_WIDTH
#IMAGE_DEPTH = 3


yaml_text = open("app.yaml", 'r')
content = load(yaml_text, Loader=Loader)

DATA_DIR = pathlib.Path(content["DATA_DIR"])
MODELS_DIR = content["MODELS_DIR"]

TARGET_NAME = content["TARGET_NAME"]

IMAGE_WIDTH = content["IMAGE_WIDTH"]
IMAGE_HEIGHT = content["IMAGE_HEIGHT"]
IMAGE_DEPTH = content["IMAGE_DEPTH"]


def load_image(path):
    """Load an image as numpy array
    """
    return plt.imread(path)


def predict_image(path, model):
    """Predict plane identification from image.
    
    Parameters
    ----------
    path (Path): path to image to identify
    model (keras.models): Keras model to be used for prediction
    
    Returns
    -------
    Predicted class
    """
    images = np.array([np.array(Image.open(path).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))])
    print(images.shape)
    prediction_vector = model.predict(images)
    predicted_classes = np.argmax(prediction_vector, axis=1)
    probability = prediction_vector[0][predicted_classes] * 100
    return predicted_classes[0], probability, prediction_vector


def load_model(path):
    """Load tf/Keras model for prediction
    """
    return tf.keras.models.load_model(path)
    

model = load_model('models/manufacturer.h5')
model.summary()



st.title("Identification d'avion")

uploaded_file = st.file_uploader("Charger une image d'avion") #, accept_multiple_files=True)#

if uploaded_file:
    loaded_image = load_image(uploaded_file)
    st.image(loaded_image)

predict_btn = st.button("Identifier", disabled=(uploaded_file is None))
if predict_btn:
    prediction, probability, prediction_vector = predict_image(uploaded_file, model)
    st.write(f"C'est un : {prediction}\n")
    st.write(f"Avec une probabilité de {probability} %")
    st.bar_chart(prediction_vector.T)
st.snow()
```
## -----app.yaml-----
```
#chemin d'accès à la base de données 
DATA_DIR : 'C:\Users\flora\Documents\planes\data'
#chemin d'accès au modèle 
MODELS_DIR : 'C:\Users\flora\Documents\planes\app\models'

TARGET_NAME : 'manufacturer'
IMAGE_WIDTH : 128
IMAGE_HEIGHT : 128
IMAGE_DEPTH : 3
```
