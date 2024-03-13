# from flask import Flask
import streamlit as st
from PIL import Image
import os
import pickle
import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
model1=pickle.load(open('mobile_model.pkl','rb'))
preprocess_input=pickle.load(open('preprocess.pkl','rb'))

def process_image(image):

    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    prediction = model1.predict(image)
    prediction = tf.nn.sigmoid(prediction)
    prediction = tf.where(prediction < 0.5, 0, 1)
    i=prediction.numpy()[0][0]
    l=['Cataract','Normal']
    

    
    return l[i]
def main():
    st.title("Eye Disease detection")
    st.sidebar.title("Upload images")
    os.makedirs("uploaded_images", exist_ok=True)
    # File uploader for image input
    uploaded_image = st.sidebar.file_uploader("Here", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Button to start processing
        if st.sidebar.button("Process"):
            # Process the image
            image_path = os.path.join("uploaded_images", "temp_image.jpg")
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getvalue())
            image = load_img(image_path, target_size=(224, 224))
            processed_image = process_image(image)
            
            # Display the processed image
            st.write("Result:",processed_image)
    
if __name__ == "__main__":
    main()
           