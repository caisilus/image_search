import streamlit as st
from PIL import Image, ImageOps

neighbors_number = 5

def main_page():
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp", "tiff"])
    if uploaded_file is None:
        return
    image = Image.open(uploaded_file).convert("RGB")
    image = ImageOps.exif_transpose(image)
    st.image(image, width=300, output_format='auto')

    st.divider()

    similar_images = [image] * neighbors_number # TODO replace with actual data
    st.title('Similar images:')
    cols = st.columns(neighbors_number)
    for i in range(neighbors_number):
        with cols[i]:
            st.image(similar_images[i])


main_page()