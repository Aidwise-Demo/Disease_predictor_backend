# Importing libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
# Loading the image
# image = plt.imread('dataset/normal/104.jpeg')


def CovidPrediction(Path):
    img = cv2.imread(Path)

    RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    RGBImg = cv2.resize(RGBImg, (224, 224))
    # plt.imshow(RGBImg)

    image = np.array(RGBImg) / 255.0

    model = tf.keras.models.load_model('covidmodel')

    # Making predictions on the pre-processed image
    predictions = model.predict(np.array([image]))

    # Printing the predictions
    # print(predictions)

    per = np.argmax(predictions, axis=1)
    if per == 1:
        return 'Covid Detected'
    else:
        return 'Covid Not Detected'