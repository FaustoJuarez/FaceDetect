# Import required libraries
import PIL
import streamlit as st
from ultralytics import YOLO
import numpy as np

# Replace the relative path to your weight file
model_path = 'weights/best.pt'

# Setting page layout
st.set_page_config(
    page_title="Deteccion de Rostros",
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
st.title("Deteccion de Rostros con YOLOv8")

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
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(uploaded_image,
                 caption="Imagen cargada",
                 use_column_width=True
                 )
    
    if st.sidebar.button('Detectar insectos'):
        # Convert PIL Image to numpy array
        img_array = np.array(uploaded_image)
        
        # Perform prediction
        res = model.predict(img_array, conf=confidence)
        
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        
        with col2:
            st.image(res_plotted,
                     caption='Imagen detectada',
                     use_column_width=True
                     )
