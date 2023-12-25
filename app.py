import streamlit as st
import PIL

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp", "tiff"])
if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file).convert("RGB")
    image = PIL.ImageOps.exif_transpose(image)