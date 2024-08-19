import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import pandas as pd
import os

# URL of the .tflite file in your GitHub repository
url = "https://raw.githubusercontent.com/ryank23331682/Deep_Learning_Skin_Classification/f62bae1e7ec8cf7a59f52a0f9f0285c5b7263395/mobilenetv2_modelV2.tflite"

# Local path where the model will be saved
local_path = "mobilenetv2_modelV2.tflite"

# Download the file from the GitHub repository if it does not exist locally
if not os.path.exists(local_path):
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful
    with open(local_path, 'wb') as f:
        f.write(response.content)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=local_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels with full words and descriptions
class_info = {
    'akiec': {'full_word': 'Actinic Keratosis', 'description': 'A precancerous skin condition characterized by rough, scaly patches on sun-exposed skin.'},
    'bcc': {'full_word': 'Basal Cell Carcinoma', 'description': 'A type of skin cancer that begins in the basal cells of the skin.'},
    'bkl': {'full_word': 'Benign Keratosis', 'description': 'A non-cancerous skin growth, often referred to as a seborrheic keratosis.'},
    'df': {'full_word': 'Dermatofibroma', 'description': 'A common benign skin tumor that is usually firm and raised.'},
    'mel': {'full_word': 'Melanoma', 'description': 'A serious form of skin cancer that begins in the cells that produce pigment (melanocytes).'},
    'nv': {'full_word': 'Nevus', 'description': 'A common skin lesion also known as a mole, which can be benign or a precursor to melanoma.'},
    'vasc': {'full_word': 'Vascular Lesion', 'description': 'A type of skin lesion that involves abnormal blood vessels, such as hemangiomas.'}
}
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

    # Get the predicted class index and probabilities
    predicted_probabilities = output_data[0]
    predicted_class_index = np.argmax(predicted_probabilities)
    
    # Create a dictionary of class labels with their corresponding probabilities
    class_probabilities = {class_labels[i]: predicted_probabilities[i] for i in range(len(class_labels))}
    
    # Get the corresponding class label
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label, class_probabilities


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
    class_label, class_probabilities = predict(image)
    # Display the predicted class label
    st.write(f"Predicted Class: {class_info[class_label]['full_word']}")
    
    # Convert the class probabilities dictionary to a DataFrame
    class_probabilities_df = pd.DataFrame(list(class_probabilities.items()), columns=['Class Abbreviation', 'Probability'])
    
    # Add Full Word and Description columns
    class_probabilities_df['Full Word'] = class_probabilities_df['Class Abbreviation'].map(lambda x: class_info[x]['full_word'])
    class_probabilities_df['Description'] = class_probabilities_df['Class Abbreviation'].map(lambda x: class_info[x]['description'])
    
    # Reorder columns for better readability
    class_probabilities_df = class_probabilities_df[['Class Abbreviation', 'Full Word', 'Description', 'Probability']]
    
    # Display the class probabilities in a table without index
    st.write("Class Probabilities:")
    st.write(class_probabilities_df.to_html(index=False), unsafe_allow_html=True)
