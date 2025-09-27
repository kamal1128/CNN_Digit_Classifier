import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ----------------------------
# Load CNN Model
# ----------------------------
@st.cache_resource
def load_cnn_model():
    model = load_model("model/cnn_model.h5")
    return model

cnn_model = load_cnn_model()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Digit Classifier", layout="wide")

# Centered title
st.markdown("<h1 style='text-align: center;'>🖊️ Handwritten Digit Classifier (0–9)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Draw a digit inside the canvas and let the CNN model predict it.</p>", unsafe_allow_html=True)

# ----------------------------
# Layout
# ----------------------------
col1, col2, col3 = st.columns([1,2,1])  # Center canvas in middle column

with col2:
    # Canvas
    st.markdown("<h1 style='text-align: center;'>✏️ Draw Here</h1>", unsafe_allow_html=True)
    canvas_result = st_canvas(
        fill_color="#000000",       # Black ink
        stroke_width=10,            # Brush thickness
        stroke_color="#FFFFFF",     # White ink
        background_color="#000000", # Black canvas
        update_streamlit=True,
        height=550,
        width=800,
        drawing_mode="freedraw",
        key="canvas"
    )

    # Prediction
    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype("uint8")

        # Convert RGBA → grayscale
        img_pil = Image.fromarray(img).convert("L")

        # Resize to MNIST shape (28x28)
        img_resized = img_pil.resize((28, 28))

        # Normalize and reshape for CNN
        img_array = np.array(img_resized) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        prediction = cnn_model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        # Display prediction
        st.markdown("<h3 style='text-align: center; color: #FF4B4B;'>🧠 Model Prediction</h3>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>The digit is: {predicted_digit}</h2>", unsafe_allow_html=True)

        # Show processed image
        st.image(img_resized, caption="Processed Image (28x28)", width=120, use_container_width=False)
