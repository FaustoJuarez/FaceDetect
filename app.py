import PIL
import streamlit as st
from ultralytics import YOLO
import numpy as np
import logging
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO)

# Replace the relative path to your weight file
model_path = 'weights/best.pt'

# Setting page layout
st.set_page_config(
    page_title="Deteccion de Insectos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Creating sidebar
with st.sidebar:
    st.header("Cargue su imagen")
    source_img = st.file_uploader(
        "Imagen", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    confidence = 0.15

# Creating main page heading
st.title("Deteccion de Insectos con YOLOv8")

# Creating two columns on the main page
col1, col2 = st.columns(2)

# Load the model
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Adding image to the first column if image is uploaded
if source_img:
    with col1:
        try:
            # Opening the uploaded image
            uploaded_image = PIL.Image.open(source_img)
            # Adding the uploaded image to the page with a caption
            st.image(uploaded_image,
                     caption="Imagen cargada",
                     use_column_width=True
                     )
        except Exception as ex:
            st.error("Error opening the image.")
            st.error(ex)
            logging.error(f"Error opening image: {ex}")
    
    if st.sidebar.button('Detectar insectos'):
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(uploaded_image)
            
            # Convert RGB to BGR (OpenCV uses BGR)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Log the type and shape of the image
            logging.info(f"Image type: {type(img_bgr)}, Shape: {img_bgr.shape}")
            
            # Perform prediction
            results = model(img_bgr, conf=confidence)
            
            # Plot the results
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = PIL.Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            
            with col2:
                st.image(im,
                         caption='Imagen detectada',
                         use_column_width=True
                         )
        except Exception as ex:
            st.error("Error during insect detection.")
            st.error(ex)
            logging.error(f"Error during detection: {ex}")
