import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# URL of the .tflite file in your GitHub repository
url = "https://raw.githubusercontent.com/ryank23331682/Deep_Learning_Skin_Classification.git/master/mobilenetv2_model.tflite"

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="https://raw.githubusercontent.com/ryank23331682/Deep_Learning_Skin_Classification.git/master/mobilenetv2_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels
class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def predict(image):
    # Preprocess the image to required size and cast
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    # Set the tensor to the input data
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run the interpreter
    interpreter.invoke()

    # The function `get_tensor` returns a copy of the tensor data
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class index
    predicted_class_index = np.argmax(output_data[0])
    
    # Get the corresponding class label
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label, output_data

st.title("Deep Learning for Skin Cancer Detection")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file)

    # Display the image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict the image
    class_label, predictions = predict(image)
    # Display the predicted class label and raw predictions
    st.write(f"Predicted Class: {class_label}")
    st.write("Raw Predictions:")
    st.write(predictions)
