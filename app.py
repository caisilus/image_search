import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import os
from sklearn.cluster import KMeans
import pickle
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
import random

def get_image_descriptors(gray):
    _, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

def get_descriptors_from_dir(dir):
    descriptors = []
    for filename in os.listdir(dir):
        image_path = os.path.join(dir, filename)
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        image_descriptors = get_image_descriptors(image)

        if image_descriptors is None or image_descriptors.ndim == 1:
            continue

        descriptors.append(image_descriptors)
    return descriptors

def get_normalized_hist(clusters):
    hist_data = np.zeros((n_clusters))
    for cluster in clusters:
        hist_data[cluster] += 1

    hist_data /= clusters.shape[0]

    return hist_data

def vectorize_image_from_file(model, image_file):
    image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
    descriptors = get_image_descriptors(image)

    if descriptors is None:
        return None
    if descriptors.ndim == 1:
        descriptors = descriptors.reshape(1, -1)

    clusters = model.predict(descriptors)
    hist_data = get_normalized_hist(clusters)
    return hist_data

def find_similar(model, image_filename, number):
    input_vector = vectorize_image_from_file(model, image_filename)
    distances, indices = model.kneighbors([input_vector], n_neighbors=number)
    pics_pathes = data.loc[indices[0]]["image_path"]
    return [Image.open(image_path).convert("RGB") for image_path in pics_pathes]


def main_page():
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp", "tiff"])
    if uploaded_file is None:
        return
    image = Image.open(uploaded_file).convert("RGB")
    image = ImageOps.exif_transpose(image)
    st.image(image, width=300, output_format='auto')

    st.divider()

    similar_images = find_similar(neighbours_model, uploaded_file, neighbors_number)
    st.title('Similar images:')
    cols = st.columns(neighbors_number)
    for i in range(neighbors_number):
        with cols[i]:
            st.image(similar_images[i])


sift = cv.SIFT_create()
n_clusters = 2048
neighbors_number = 10

data = pd.read_csv()
X = np.vstack(data["vector"].values)

neighbours_model = NearestNeighbors(metric='cosine', algorithm='brute')
neighbours_model.fit(X)

main_page()