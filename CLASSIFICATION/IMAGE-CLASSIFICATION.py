import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from google.colab import files
import os

# Load the model from Pickle
with open('cnn_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model loaded successfully!")

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Upload image from local machine (Colab-compatible)
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Verify that the file exists
if not os.path.exists(image_path):
    print(f"File not found: {image_path}")
else:
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(32, 32))  # Resize to CIFAR-10 size
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    # Display the image and predicted class
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_class}')
    plt.axis('off')
    plt.show()
