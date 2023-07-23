import typing
import string

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow import keras
from keras import datasets, layers, models
import numpy as np
import cv2
import os
import hashlib

import urllib.request
import urllib.parse

import keras_ocr

from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import random


def recolor(img, width, threshold):
    fn = lambda x : 255 if x < threshold else 0
    img_recolored = img.convert('L').point(fn, mode='1').resize((width, 28))
    return img_recolored

def classify(img, threshold):
    aspect_ratio = img.size[0] / img.size[1]
    resize_width = 0
    if aspect_ratio * 28 < 50:
        resize_width = 50
    elif aspect_ratio * 28 > 149:
        resize_width = 149
    else:
        resize_width = round(aspect_ratio * 28)

    recolor(img, resize_width, threshold)
    converted_img = np.array(recolor(img, resize_width, threshold))
    converted_img = np.array([[[y] for y in converted_img[x]] for x in range(len(converted_img))])

    print("Resize width: " + str(resize_width) + "px")

    json_file = open("model/model" + str(resize_width - 50) + ".json", "r")
    model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(model_json)
    loaded_model.load_weights("model/model" + str(resize_width - 50) + ".h5")
    loaded_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    test_loss, test_acc = loaded_model.evaluate(np.array([converted_img]), np.array([1]), verbose=2)
    print(test_acc)
    if test_acc == 0:
        print(True)
        return "According to our model's analysis of our training set, you are likely to have dyslexia. The results of this test are based on correlational data, we persuade you to consult a medical professional if you require a definitive diagnosis. For Further information look at our resources page. "
    else:
        return False
    
def predict(filename, threshold=140):
    input = Image.open(filename)
    matplotlib.pyplot.switch_backend('Agg')
    images = [keras_ocr.tools.read(img) for img in [filename]]
    pipeline = keras_ocr.pipeline.Pipeline()
    prediction_groups = pipeline.recognize(images)
    keras_ocr.tools.drawAnnotations(image=images[0], predictions=prediction_groups[0])
    im_cropped = input.crop((prediction_groups[0][0][1][0][0], prediction_groups[0][0][1][0][1], prediction_groups[0][0][1][2][0], prediction_groups[0][0][1][2][1]))
    
    return classify(im_cropped, threshold)
