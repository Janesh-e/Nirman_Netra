
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2

# Step 2: Load and preprocess the input image
def preprocess_image(image_path, target_size=(640, 640)):
    # Load the image
    img = load_img(image_path, target_size=target_size, color_mode="rgb")
    # Convert the image to a numpy array and normalize it (if needed)
    img_array = img_to_array(img) / 255.0
    # Add batch dimension (since the model expects input shape [batch_size, height, width, channels])
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

