import PIL
import streamlit as st
from ultralytics import YOLO
import numpy as np
import logging

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
            # Convert PIL Image to RGB mode
            rgb_image = uploaded_image.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(rgb_image)
            
            # Log the type and shape of the image
            logging.info(f"Image type: {type(img_array)}, Shape: {img_array.shape}")
            
            # Perform prediction
            res = model(img_array, conf=confidence)
            
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            
            with col2:
                st.image(res_plotted,
                         caption='Imagen detectada',
                         use_column_width=True
                         )
        except Exception as ex:
            st.error("Error during insect detection.")
            st.error(ex)
            logging.error(f"Error during detection: {ex}")
