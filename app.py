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
import torch
import clip

def load_model_from(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    return model

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

def load_image_for_clip(image_filename):
    return preprocess(Image.open(image_filename)).unsqueeze(0).to(device)
def encode_with_clip(model, image_filename):
    image = load_image_for_clip(image_filename)
    image_features = model.encode_image(image)
    image_features = image_features.cpu().detach().numpy().squeeze().astype(np.float64)
    return image_features

def find_similar(model_type, image_filename, number):
    indices = []
    if model_type == "KMeans":
        input_vector = vectorize_image_from_file(kmeans_model, image_filename)
        distances, indices = kmeans_neighbours.kneighbors([input_vector], n_neighbors=number)
    else:
        indices = encode_with_clip(clip_model, image_filename)

    pics_pathes = data.loc[indices[0]]["image_path"]
    return [Image.open(image_path).convert("RGB") for image_path in pics_pathes]


def main_page():
    model_type = st.radio("Choose model",["KMeans", "CLIP"])

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp", "tiff"])
    if uploaded_file is None:
        return
    image = Image.open(uploaded_file).convert("RGB")
    image = ImageOps.exif_transpose(image)
    st.image(image, width=300, output_format='auto')

    st.divider()

    similar_images = find_similar(model_type, uploaded_file.name, neighbors_number)
    # similar_images = [image] * neighbors_number
    st.title('Similar images:')
    row_num = 2
    col_num = neighbors_number // row_num
    rows = []
    for i in range(row_num):
        rows.append(st.columns(col_num))

    for i in range(row_num):
        for j in range(col_num):
                with rows[i][j]:
                    st.image(similar_images[i * col_num + j])


sift = cv.SIFT_create()
n_clusters = 2048
neighbors_number = 10

def init_kmeans():
    data = pd.read_csv('image_database.csv')
    X = np.vstack(data["vector"].values).astype(np.object0)
    kmeans_model = load_model_from('kmeans.pickle')
    arr_X = [None] * X.shape[0]
    for i in range(X.shape[0]):
        arr_X[i] = (np.fromstring(X[i][0][1:-1], sep=', '))
    kmeans_neighbours = NearestNeighbors(metric='cosine', algorithm='brute')
    kmeans_neighbours.fit(arr_X)
    return kmeans_model, kmeans_neighbours, data

def init_clip():
    data = pd.read_csv('image_database_clip.csv')
    X = np.vstack(data["vector"].values)
    clip_neighbours = NearestNeighbors(metric='cosine', algorithm='brute').astype(np.object0)
    clip_neighbours.fit(X)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return clip_neighbours, model, preprocess, device


kmeans_model, kmeans_neighbours, data = init_kmeans()

clip_neighbours, clip_model, preprocess, device = init_clip(data)

main_page()