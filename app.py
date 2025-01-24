import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("artifacts/facial_expression_model.h5")

# Define class names (make sure these match your training data)
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] # 0-6

# Set up the Streamlit app
st.set_page_config(page_title="Facial Expression Recognition", layout="wide")
st.title("Facial Expression Recognition App")
st.write("Upload an image and the model will predict the facial expression.")

# Sidebar for file upload
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(uploaded_file):
    """Preprocess the uploaded image for prediction."""
    img = Image.open(uploaded_file)
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((48, 48))  # Resize to match the model's input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1) # Add channel dimension
    return img_array

def display_prediction(prediction):
    """Display the prediction results."""
    predicted_class = np.argmax(prediction)
    st.success("Prediction complete!")
    st.write(f"**Predicted Expression:** {class_names[predicted_class]}")
    st.write("### Prediction Probabilities")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob:.2f}")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image and make prediction
    with st.spinner('Processing image...'):
        img_array = preprocess_image(uploaded_file)
        prediction = model.predict(img_array)

    # Display the results
    display_prediction(prediction)
else:
    st.info("Please upload an image to get started.")
