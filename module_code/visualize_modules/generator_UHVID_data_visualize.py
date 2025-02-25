
######################################

#Visualization for Generating uhvid
#####################################

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Input, LSTM, Dense, Reshape, GRU, Dropout
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing import image
import cv2
from multiprocessing import Pool, cpu_count
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import distance as dist
import argparse
import glob
import datetime
import re
import math
import statistics
from skimage import measure
import pandas as pd
import csv
import json
from sklearn.metrics import mean_squared_error
import scipy.misc
import sys
import pickle
import time
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances_argmin_min
from PIL import UnidentifiedImageError
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
from moviepy.video.io.VideoFileClip import VideoFileClip 


class GeneratorUHVIDdataVisualize:

    def __init__(self):

      self.frame = []
      self.video_frames = []
      self.feature_maps = []
      self.features_data = []
      self.centroid_frames = []
      self.centroids = []
      self.imageframe = []
      self.closest_frames = []
      self.similarity_matrix = []
      self.unique_labels = []
      self.noise_indices = []
      self.labels = []
      self.features = []
      self.imageframe = []
      self.closest_frames_indices = []
      self.closest_frames_features_data = []
      self.frame_features = []
      self.closeset_frames_features_data_for_lstm = []
      self.closeset_frames_features_data_for_gr = []
      self.video_id_features = []
      self.video_id_lstm = []
      self.video_id_features_lstm = []
      self.video_id_gru = []
      self.video_id_features_gru = []
      self.video_id_size_gru = 0.0
      self.video_id_size_lstm = 0.0
      self.video = " "
      self.eps = 0.0


    def extract_frames(self, video):

      cap = cv2.VideoCapture(video)
      frame_shape = []
      frames = []
      imageframe = []
      while cap.isOpened():
          ret, frame = cap.read()
          if not ret:
              break
          # Convert BGR (OpenCV format) to RGB
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          # Resize the frame to 224x224 as expected by VGG16
          frame = cv2.resize(frame, (224, 224))
          frame_shape = frame
          frames.append(frame)

      cap.release()
      return frames, imageframe


    def extract_features(self, video_frames):

      feature_maps = []
      features = []
      # Load pre-trained CNN models (VGG16 here)
      base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
      cnn_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
      #cnn_model.summary()

      # Check if frames are extracted
      if not video_frames:
          raise ValueError("No frames extracted from the video.")

      # Preprocess frames
      video_frames = [preprocess_input(np.expand_dims(frame, axis=0)) for frame in video_frames]

      # List to hold feature maps for all frames


      for frame in video_frames:
          # Extract features
          features = cnn_model.predict(frame, verbose=0)
          # Append the feature map to the list
          feature_maps.append(features)

      # Check if feature maps are extracted
      if not feature_maps:
          raise ValueError("No feature maps extracted from the frames.")

      # Check if feature maps list is not empty
      if feature_maps:
          # Stack all feature maps along the first dimension (frames dimension)
          combined_features = np.vstack(feature_maps)

          # Perform global average pooling on the combined features
          features = np.mean(combined_features, axis=(1, 2))  # Average pooling over height and width

          # The final pooled_features is a 2D array of shape (n_frames, n_channels)
          # If you need a single feature vector for all frames, you can average pool over the frames as well

      return features, feature_maps



    def keyframes_extracted(self, closest_frames, imageframe):


      keyframes = []

      for element in closest_frames:
        keyframes.append(imageframe[element])

      keyframes = np.array(keyframes)

      # Make sure all the images have the same dimensions
      # Calculate the number of rows and columns for the grid
      num_rows = 3  # Specify the number of rows you want
      num_cols = 4  # Specify the number of columns you want


      # Get the total number of image data available
      total_images = len(keyframes)

      # Create a figure and axis objects
      fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 9))

      # Flatten the 2D axs array to simplify indexing
      axs = axs.ravel()

      # Plot each image in the grid
      for i in range(min(total_images, num_rows*num_cols)):
          axs[i].imshow(self.frames[i])
          axs[i].axis('off')  # Turn off axis labels
          axs[i].set_title(f"Frame {i+1}")

      # If there are empty slots in the grid, hide them
      for i in range(total_images, num_rows*num_cols):
          axs[i].axis('off')

      # Adjust the spacing between subplots for better visualization
      plt.tight_layout()

      # Show the plot
      plt.show()




    def dbscan_fit(self, features_data, eps, min_samples, sigma):

      perplexity=30
      n_samples = features_data.shape[0]
      if perplexity >= n_samples:
        perplexity = GeneratorUHVIDdataVisualize.find_best_perplexity_kl_divergence(features_data)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        features_data = tsne.fit_transform(features_data)
      else:
        tsne = TSNE(n_components=2, random_state=42)
        features_data = tsne.fit_transform(features_data)



      unique_labels, noise_indices, labels, centroids, closest_frames = [],[],[],[],[]

      # Perform DBSCAN on the similarity matrix
      dbscan = DBSCAN(eps=eps, min_samples=min_samples)
      labels = dbscan.fit_predict(features_data)

      noise_indices = np.where(labels == -1)[0]
      unique_labels = np.unique(labels)

      # Calculate centroids for each cluster
      centroids = []
      for label in unique_labels:
        if label != -1:  # Ignore noise
              cluster_points = features_data[labels == label]
              centroid = cluster_points.mean(axis=0)
              centroids.append(centroid)
      centroids = np.array(centroids)

      # Find the closest frame to each centroid
      closest_frames = []
      for centroid in centroids:
          distances = cdist(features_data, [centroid])
          closest_frame_index = np.argmin(distances)
          closest_frames.append(closest_frame_index)
      closest_frames = np.array(closest_frames)

      #Plot DBSCAN clusters ###############temporary########

      ###------------
      plt.scatter(features_data[:, 0], features_data[:, 1], c=labels)
      plt.xlabel('t-SNE Component 1')
      plt.ylabel('t-SNE Component 2')
      plt.title('DBSCAN Clusters on t-SNE Transformed Data')
      plt.show()

      ###------------



      return unique_labels, noise_indices, labels, centroids, closest_frames



    #Retrieve the closest frame to each centroid
    def get_closest_frames(self, features_data, centroids):

      closest_frames_indices = []
      closest_frames_features_data = []

      perplexity=30
      n_samples = features_data.shape[0]
      if perplexity >= n_samples:
        perplexity = GeneratorUHVIDdataVisualize.find_best_perplexity_kl_divergence(features_data)

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        features_data = tsne.fit_transform(features_data)
      else:
        tsne = TSNE(n_components=2, random_state=42)
        features_data = tsne.fit_transform(features_data)

      for centroid in centroids:
          distances = cdist(features_data, [centroid])
          closest_frame_index = np.argmin(distances)
          closest_frames_indices.append(closest_frame_index)
          closest_frames_features_data.append(features_data[closest_frame_index])

      return closest_frames_indices, np.asarray(closest_frames_features_data)

    # Retrieve the video frames corresponding to the centroids
    def retrieve_video_frames(self, video_frames, closest_frames):

      return [video_frames[idx] for idx in closest_frames]

    def retrieve_centroid_features(self, closest_frames, features_data):

      centroid_features = features_data
      return [centroid_features[idx] for idx in closest_frames]


    def plot_culsters2(self, features_data, unique_labels, labels):
      perplexity=30
      n_samples = features_data.shape[0]
      if perplexity >= n_samples:
        perplexity = GeneratorUHVIDdataVisualize.find_best_perplexity_kl_divergence(features_data)
        #print(f"perplexity is : {perplexity}")

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_transformed = tsne.fit_transform(features_data)
      else:
        tsne = TSNE(n_components=2, random_state=42)
        tsne_transformed = tsne.fit_transform(features_data)

      centroids_2d = self.centroids
      plt.figure(figsize=(10, 5))
      for label in  unique_labels:
          if label == -1:
              # Plot noise points in black
              plt.scatter(tsne_transformed[ labels == label][:, 0], tsne_transformed[labels == label][:, 1], marker='o', s=50, color='black', label='Noise')
          else:
              # Plot clustered points with different colors
              plt.scatter(tsne_transformed[labels == label][:, 0], #tsne_transformed[ labels == label][:, 1], marker='o', s=50) ###, label=f'Cluster {label}'###
              tsne_transformed[ labels == label][:, 1], marker='o', s=50, label=f'Cluster {label}') ###

      # Plot centroids
      plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], marker='x', s=100, color='red', label='Centroids')
      plt.xlabel('t-SNE Component 1')
      plt.ylabel('t-SNE Component 2')
      plt.legend()
      plt.show()


    def plot_frames_grid(self, centroid_frames):

      # Make sure all the images have the same dimensions
      # Get the total number of image data available
      num_rows = 0

      total_images = len(centroid_frames)
      # Ensure data is in the valid range for imshow
      centroid_frames = [np.clip(frame, 0, 1) if frame.dtype == np.float32 else np.clip(frame, 0, 255) for frame in centroid_frames]
      # Calculate the number of rows and columns for the grid
      # Calculate the number of rows and columns for the grid
      num_cols = 5  # Specify the number of columns you want
      num_rows = (total_images + num_cols - 1) // num_cols  # Calculate the number of rows

      # Create a figure and axis objects
      fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 9))

      # Flatten the 2D axs array to simplify indexing
      axs = axs.ravel()

      # Plot each image in the grid
      for i in range(total_images):
          axs[i].imshow(centroid_frames[i])
          axs[i].axis('off')  # Turn off axis labels
          axs[i].set_title(f"Frame {i+1}")

      # If there are empty slots in the grid, hide them
      for i in range(total_images, num_rows*num_cols):
          axs[i].axis('off')

      # Adjust the spacing between subplots for better visualization
      plt.tight_layout()

      # Show the plot
      plt.show()

      # Print the total number of image data available
      print(f"Total number of image data: {total_images}")

    def find_best_perplexity_kl_divergence(features_data):
      n_samples = features_data.shape[0]
      perplexity_values = [5, 10, 20, 30, 40, 50]
      best_kl_divergence = float('inf')



      # Filter perplexity values to be less than the number of samples
      perplexity_values = [p for p in perplexity_values if p < n_samples]

      for perplexity in perplexity_values:
          tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
          tsne_transformed = tsne.fit_transform(features_data)

          kl_divergence = tsne.kl_divergence_

          if kl_divergence < best_kl_divergence:
              best_kl_divergence = kl_divergence
              best_perplexity = perplexity

      return best_perplexity

    def determine_eps_from_graph(self, features_data):


      perplexity=30
      n_samples = features_data.shape[0]
      if perplexity >= n_samples:
        perplexity = GeneratorUHVIDdataVisualize.find_best_perplexity_kl_divergence(features_data)

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        features_data = tsne.fit_transform(features_data)
      else:
        tsne = TSNE(n_components=2, random_state=42)
        features_data = tsne.fit_transform(features_data)

      # Determine the value of k (minPts)
      k = 3  # This is typically chosen based on domain knowledge or by experimentation

      # Fit the NearestNeighbors model
      neigh = NearestNeighbors(n_neighbors=k)
      nbrs = neigh.fit(features_data)
      distances, indices = nbrs.kneighbors(features_data)

      # Sort distances to the k-th nearest neighbor
      distances = np.sort(distances[:, k-1], axis=0)

      tsne = TSNE(n_components=2, random_state=42)

      # Identify the elbow point using KneeLocator
      kneedle = KneeLocator(range(len(distances)), distances, S=1, curve="convex", direction="increasing")

      # The epsilon value is the distance at the knee point

      eps = distances[kneedle.elbow]
      print(f'Optimal epsilon (eps) value: {eps}') ###
      ############################################# REUIRED
      #Plot to visualize (optional)
      plt.plot(distances)
      plt.axvline(x=kneedle.knee, color='r', linestyle='--', label='eps')

      plt.xlabel('Points sorted by distance')
      plt.ylabel('k-distance')
      plt.legend()
      plt.show()
      ################################################
      return eps


    def generate_lstm_csv(self, closeset_frames_features_data_for_lstm, video_id_features, video_ids):

      # Reshape the arrays to single rows
      closest_frame_features_reshaped = closeset_frames_features_data_for_lstm.reshape(1, -1).tolist()[0]
      video_id_features_reshaped = video_id_features.reshape(1, -1).tolist()[0]
      video_ids_reshaped = video_ids.reshape(1, -1).tolist()[0]
      dataFrame = {
          'closest_frame_features': [closest_frame_features_reshaped],
          'video_id_features': [video_id_features_reshaped]
          ,'video_ids': [video_ids_reshaped]

      }

      delimiter = ','
      datafeatureCSVname = "CSV_datafeatures.csv"
      fileExists = os.path.isfile(datafeatureCSVname)

      with open(datafeatureCSVname, 'a', newline='') as CSVFile:
          csvHeader = ['closest_frame_features', 'video_id_features', 'video_ids']
          writer = csv.DictWriter(CSVFile, fieldnames=csvHeader)
          if not fileExists:
              writer.writeheader()
          # Write the data
          writer.writerow({
              'closest_frame_features': json.dumps(dataFrame['closest_frame_features']),
              'video_id_features': json.dumps(dataFrame['video_id_features'])
              ,'video_ids': json.dumps(dataFrame['video_ids'])
          })

      print(f"Data successfully written to {datafeatureCSVname}")

    def create_lstm_model(self, batch_size, input_shape):
      model = Sequential()
      model.add(LSTM(256, return_sequences=True, batch_input_shape=(batch_size, *input_shape), stateful=True))
      model.add(LSTM(128, return_sequences=False, stateful=True))
      model.add(Dense(64, activation='sigmoid'))
      return model


    def create_gru_model(self, batch_size, input_shape):
      model = Sequential()
      model.add(GRU(256, return_sequences=True, input_shape= input_shape))
      model.add(GRU(128, return_sequences=True))
      model.add(GRU(64, activation='relu', return_sequences=False))
      model.add(Dense(64, activation='sigmoid'))

      return model

    def lstm_generate_video_id(self, closeset_frames_features_data_for_lstm):


      # Define the input shape
      closeset_frames_features_data_for_lstm = np.expand_dims(closeset_frames_features_data_for_lstm, axis=0)
      batch_size = closeset_frames_features_data_for_lstm.shape[0]
      print("batch_size") ###
      print(batch_size) ###
      input_shape = (closeset_frames_features_data_for_lstm.shape[1], closeset_frames_features_data_for_lstm.shape[2])
      inputs = Input(shape=input_shape)
      print("Expanded size of clossest frames") ###
      print(closeset_frames_features_data_for_lstm.shape) ###

      # Define the LSTM layers
      x = LSTM(256, return_sequences=True)(inputs)
      x = LSTM(128, return_sequences=False)(x)

      # Define the feature layer
      feature_layer = Dense(64, activation='sigmoid')(x)

      # Create the model
      base_model_lstm = self.create_lstm_model(batch_size, input_shape)
      #base_model_lstm.summary()

      # Get the features
      lstm_video_id_features = []
      lstm_video_id_features = base_model_lstm.predict(closeset_frames_features_data_for_lstm, verbose=0)
      print("Feature vector shape:", lstm_video_id_features.shape)
      return lstm_video_id_features

    def gru_generate_video_id(self, closeset_frames_features_data_for_gru):

      # Define the input shape
      closeset_frames_features_data_for_gru = np.expand_dims(closeset_frames_features_data_for_gru, axis=0)
      batch_size = closeset_frames_features_data_for_gru.shape[0]
      print("batch_size")  ###
      print(batch_size)  ###
      input_shape = (closeset_frames_features_data_for_gru.shape[1], closeset_frames_features_data_for_gru.shape[2])
      inputs = Input(shape=input_shape)
      print("Expanded size of clossest frames")
      print(closeset_frames_features_data_for_gru.shape)

      # Define the GRU layers
      x = GRU(256, return_sequences=True)(inputs)
      x = GRU(128, return_sequences=True)(x)
      x = GRU(64, activation='relu', return_sequences=False)(x)
      # Define the feature layer
      feature_layer = Dense(64, activation='sigmoid')(x)

      # Create the model
      base_model_gru = self.create_gru_model(batch_size, input_shape)
      #base_model_lstm.summary()

      # Get the features
      gru_video_id_features = []
      gru_video_id_features = base_model_gru.predict(closeset_frames_features_data_for_gru, verbose=0)

      
      return gru_video_id_features


    ## UVID DATA GENERATOR
    def generatorUHVIDdataVisualize(self, video):


      self.eps = 0.0
      seed = 42
      random.seed(seed)
      np.random.seed(seed)
      self.video_frames, self.imageframe = self.extract_frames(video)
      self.features_data, self.feature_maps = self.extract_features(self.video_frames)

      print(f"Video Frames Shape: {np.array(self.video_frames).shape}") ###
      print(f"Feature Map Shape: {np.array(self.feature_maps).shape}") ###
      print(f"Features Data Shape: {self.features_data.shape}") ###

      self.eps = self.determine_eps_from_graph(self.features_data) ###

      ###########################input parameters#############################

      self.unique_labels, self.noise_indices, self.labels, self.centroids, self.closest_frames = self.dbscan_fit(self.features_data, self.eps, min_samples=3, sigma = 1 )

      from sklearn.metrics import silhouette_score, davies_bouldin_score


      silhouette = silhouette_score(self.features_data, self.labels)

      print(f'Silhouette Score: {silhouette}')
      print(f"labels : {len(self.labels)}") ###
      print(f"unique_labels : {len(self.unique_labels)}")  ###
      print(f"noise_indices: {len(self.noise_indices)}") ###
      print(f"centroids: {len(self.centroids)}") ###
      print(f"closest_frames: {len(self.closest_frames)}") ###
      warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.layers.rnn.rnn")

      self.plot_culsters2(self.features_data, self.unique_labels, self.labels)###

      self.closest_frames_indices, self.closest_frames_features_data = self.get_closest_frames(self.features_data, self.centroids)

      # Retrieve the video frames corresponding to the centroids
      self.retrieve_video_frames(self.video_frames, self.closest_frames)
      self.centroid_frames = self.retrieve_video_frames(self.video_frames, self.closest_frames)

      # Plot the retrieved frames in a grid
      self.plot_frames_grid(np.array(self.centroid_frames)) ###
      self.closeset_frames_features_data_for_gru = np.asarray(self.closest_frames_features_data)

      #print(f"closeset_frames_features_data_for_gru.shape", {self.closeset_frames_features_data_for_gru.shape}) ###

      self.video_id_features_gru = self.gru_generate_video_id(self.closeset_frames_features_data_for_gru)
      #print(self.video_id_features_gru)

      binary_hash_values = np.round(self.video_id_features_gru).astype(int)
      print(f"Generated UHVID for the video file '{video}' is: { binary_hash_values}")
      self.video_id_size_gru = sys.getsizeof(self.video_id_gru)/1024
      return binary_hash_values, self.video_id_size_gru

