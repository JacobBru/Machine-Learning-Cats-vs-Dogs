import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('cats_and_dogs_classifier.h5')

def predict_image(img_path):
    # Load and preprocess the image
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Predict the class of the image
    prediction = model.predict(img_array)

    # Print the result
    if prediction[0] > 0.5:
        print("It's a dog!")
    else:
        print("It's a cat!")

# Example usage
predict_image('predict/824.jpg')