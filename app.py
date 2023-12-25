import streamlit as st
from PIL import Image, ImageOps

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp", "tiff"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = ImageOps.exif_transpose(image)
    st.image(image, caption='Sunrise by the mountains')