import typing
import string

import tensorflow as tf
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

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import random


# normal_dataset = np.empty((39334, 28, 28))
# normal_test_dataset = np.empty((19037, 28, 28))

# def convert_image(img):
#     threshold = 127
#     fn = lambda x : 255 if x > threshold else 0
#     img_recolored = img.convert('L').point(fn, mode='1').resize((28, 28))
#     return img_recolored

# i = 0
# for file in os.listdir("data/Train/Normal"):
#     img = Image.open(r"data/Train/Normal/" + file)
#     converted = np.array(convert_image(img))
#     if converted[0][0] == 0:
#         normal_dataset[i] = converted
#         i += 1

# i = 0
# for file in os.listdir("data/Test/Normal"):
#     img = Image.open(r"data/Test/Normal/" + file)
#     converted = np.array(convert_image(img))
#     if converted[0][0] == 0:
#         normal_test_dataset[i] = converted
#         i += 1

# dyslexic_dataset = np.empty((65534, 28, 28))
# dyslexic_test_dataset = np.empty((19284, 28, 28))

# i = 0
# for file in os.listdir("data/Train/Corrected"):
#     img = Image.open(r"data/Train/Corrected/" + file)
#     converted = np.array(convert_image(img))
#     if converted[0][0] == 0:
#         dyslexic_dataset[i] = converted
#         i += 1

# i = 0
# for file in os.listdir("data/Test/Corrected"):
#     img = Image.open(r"data/Test/Corrected/" + file)
#     converted = np.array(convert_image(img))
#     if converted[0][0] == 0:
#         dyslexic_test_dataset[i] = converted
#         i += 1

# normal_dataset_reorganized = np.empty((37254, 28, 28))
# normal_test_dataset_reorganized = np.empty((19037, 28, 28))
# dyslexic_dataset_reorganized = np.empty((65534, 28, 28))
# dyslexic_test_dataset_reorganized = np.empty((19284, 28, 28))

# for i in range(37254):
#     normal_dataset_reorganized[i] = np.transpose(normal_dataset[i])

# for i in range(65534):
#     dyslexic_dataset_reorganized[i] = np.transpose(dyslexic_dataset[i])

# for i in range(19037):
#     normal_test_dataset_reorganized[i] = np.transpose(normal_test_dataset[i])

# for i in range(19284):
#     dyslexic_test_dataset_reorganized[i] = np.transpose(dyslexic_test_dataset[i])

# k = 1
# normal_dataset_list = normal_dataset_reorganized.tolist()
# for i in range(len(normal_dataset_list)):
#     k = 1
#     for j in reversed(normal_dataset_list[i]):
#         if max(j) == 0:
#             normal_dataset_list[i].pop(28 - k)
#         k += 1

# k = 1
# dyslexic_dataset_list = dyslexic_dataset_reorganized.tolist()
# for i in range(len(dyslexic_dataset_list)):
#     k = 1
#     for j in reversed(dyslexic_dataset_list[i]):
#         if max(j) == 0:
#             dyslexic_dataset_list[i].pop(28 - k)
#         k += 1

# k = 1
# normal_test_dataset_list = normal_test_dataset_reorganized.tolist()
# for i in range(len(normal_test_dataset_list)):
#     k = 1
#     for j in reversed(normal_test_dataset_list[i]):
#         if max(j) == 0:
#             normal_test_dataset_list[i].pop(28 - k)
#         k += 1

# k = 1
# dyslexic_test_dataset_list = dyslexic_test_dataset_reorganized.tolist()
# for i in range(len(dyslexic_test_dataset_list)):
#     k = 1
#     for j in reversed(dyslexic_test_dataset_list[i]):
#         if max(j) == 0:
#             dyslexic_test_dataset_list[i].pop(28 - k)
#         k += 1

# random.seed(1)

# def add_spacing(img):
#     for i in range(random.randint(0, 2)):
#         img.append([0] * 28)

# normal_training = []
# def create_training_normal(letters):
#     random.seed(letters)
#     nums = []
#     spliced_image = []
#     for i in range(letters * 20000):
#         nums.append(random.randint(0, 37253))
#     for i in range(20000):
#         for k in range(letters):
#             for j in normal_dataset_list[nums[i + 20000 * k]]:
#                 spliced_image.append(j)
#             if k != letters - 1:
#                 add_spacing(spliced_image)
#         normal_training.append(spliced_image)
#         spliced_image = []

# for i in range(3, 8):
#     create_training_normal(i)

# normal_testing = []
# def create_testing_normal(letters):
#     random.seed(letters)
#     nums = []
#     spliced_image = []
#     for i in range(letters * 1000):
#         nums.append(random.randint(0, 19036))
#     for i in range(1000):
#         for k in range(letters):
#             for j in normal_test_dataset_list[nums[i + 1000 * k]]:
#                 spliced_image.append(j)
#             if k != letters - 1:
#                 add_spacing(spliced_image)
#         normal_testing.append(spliced_image)
#         spliced_image = []

# for i in range(3, 8):
#     create_testing_normal(i)

# normal_training_cleaned = [x for x in normal_training if (len(x) >= 50 and len(x) <= 149)]
# normal_testing_cleaned = [x for x in normal_testing if (len(x) >= 50 and len(x) <= 149)]

# dyslexic_training = []
# def create_training_dyslexic(letters):
#     random.seed(letters)
#     nums = []
#     spliced_image = []
#     for i in range(letters * 20000):
#         nums.append(random.randint(0, 65533))
#     for i in range(20000):
#         for k in range(letters):
#             for j in dyslexic_dataset_list[nums[i + 20000 * k]]:
#                 spliced_image.append(j)
#             if k != letters - 1:
#                 add_spacing(spliced_image)
#         dyslexic_training.append(spliced_image)
#         spliced_image = []

# for i in range(3, 8):
#     create_training_dyslexic(i)

# dyslexic_testing = []
# def create_training_dyslexic(letters):
#     random.seed(letters)
#     nums = []
#     spliced_image = []
#     for i in range(letters * 1000):
#         nums.append(random.randint(0, 19283))
#     for i in range(1000):
#         for k in range(letters):
#             for j in dyslexic_test_dataset_list[nums[i + 1000 * k]]:
#                 spliced_image.append(j)
#             if k != letters - 1:
#                 add_spacing(spliced_image)
#         dyslexic_testing.append(spliced_image)
#         spliced_image = []

# for i in range(3, 8):
#     create_training_dyslexic(i)

# dyslexic_training_cleaned = [x for x in dyslexic_training if (len(x) >= 50 and len(x) <= 149)]
# dyslexic_testing_cleaned = [x for x in dyslexic_testing if (len(x) >= 50 and len(x) <= 149)]

# normal_training_flattened = [np.array(x).flatten().tolist() for x in normal_training_cleaned]
# normal_testing_flattened = [np.array(x).flatten().tolist() for x in normal_testing_cleaned]
# dyslexic_training_flattened = [np.array(x).flatten().tolist() for x in dyslexic_training_cleaned]
# dyslexic_testing_flattened = [np.array(x).flatten().tolist() for x in dyslexic_testing_cleaned]

# normal_training_sorted = []
# dyslexic_training_sorted = []
# model_training = []
# model_labels = []
# for i in range(50, 150):
#     normal_training_sorted.append([x for x in normal_training_flattened if len(x) / 28 == i])
#     dyslexic_training_sorted.append([x for x in dyslexic_training_flattened if len(x) / 28 == i])
#     model_training.append([x for x in normal_training_flattened if len(x) / 28 == i] + [x for x in dyslexic_training_flattened if len(x) / 28 == i])
#     model_labels.append([0] * len([x for x in normal_training_flattened if len(x) / 28 == i]) + [1] * len([x for x in dyslexic_training_flattened if len(x) / 28 == i]))

# normal_testing_sorted = []
# dyslexic_testing_sorted = []
# model_testing = []
# model_testing_labels = []
# for i in range(50, 150):
#     normal_testing_sorted.append([x for x in normal_testing_flattened if len(x) / 28 == i])
#     dyslexic_testing_sorted.append([x for x in dyslexic_testing_flattened if len(x) / 28 == i])
#     model_testing.append([x for x in normal_testing_flattened if len(x) / 28 == i] + [x for x in dyslexic_testing_flattened if len(x) / 28 == i])
#     model_testing_labels.append([0] * len([x for x in normal_testing_flattened if len(x) / 28 == i]) + [1] * len([x for x in dyslexic_testing_flattened if len(x) / 28 == i]))

# model_training = [np.array([np.array(x).reshape(28, 50 + y, 1) for x in model_training[y]]) for y in range(len(model_training))]
# model_testing = [np.array([np.array(x).reshape(28, 50 + y, 1) for x in model_testing[y]]) for y in range(len(model_testing))]

# cnns = []
# histories = []

# for i in range(100):
#     cnns.append(models.Sequential())
#     cnns[i].add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(28, i + 50, 1)))
#     cnns[i].add(layers.MaxPooling2D((2, 2)))
#     cnns[i].add(layers.Conv2D(64, (3, 3), activation='relu'))
#     cnns[i].add(layers.MaxPooling2D((2, 2)))
#     cnns[i].add(layers.Conv2D(64, (3, 3), activation='relu'))
#     cnns[i].add(layers.Flatten())
#     cnns[i].add(layers.Dense(64, activation='relu'))
#     cnns[i].add(layers.Dense(10))
#     cnns[i].summary()
#     cnns[i].compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
#     histories.append(cnns[i].fit(model_training[i], np.array(model_labels[i]), epochs=20))

# test_losses = []
# test_accs = []
# for i in range(50, 150):
#     cnns[i - 50].evaluate(model_testing[i - 50], np.array(model_testing_labels[i - 50]), verbose=2)

# def recolor(img, width, threshold):
#     fn = lambda x : 255 if x < threshold else 0
#     img_recolored = img.convert('L').point(fn, mode='1').resize((width, 28))
#     return img_recolored

# def classify(img, threshold):
#     aspect_ratio = img.size[0] / img.size[1]
#     resize_width = 0
#     if aspect_ratio * 28 < 50:
#         resize_width = 50
#     elif aspect_ratio * 28 > 149:
#         resize_width = 149
#     else:
#         resize_width = round(aspect_ratio * 28)

#     recolor(img, resize_width, threshold).show()
#     converted_img = np.array(recolor(img, resize_width, threshold))
#     converted_img = np.array([[[y] for y in converted_img[x]] for x in range(len(converted_img))])

#     test_loss, test_acc = cnns[resize_width - 50].evaluate(np.array([converted_img]), np.array([1]), verbose=2)
#     print(test_acc)
#     if test_acc == 0:
#         return False
#     else:
#         return True
    
def predict(image_file, threshold=140):
    input = Image.open(image_file)
    #images = [keras_ocr.tools.read(img) for img in [image_file]]
    #prediction_groups = pipeline.recognize(images)
    #keras_ocr.tools.drawAnnotations(image=images[0], predictions=prediction_groups[0])
    #im_cropped = input.crop((prediction_groups[0][0][1][0][0], prediction_groups[0][0][1][0][1], prediction_groups[0][0][1][2][0], prediction_groups[0][0][1][2][1]))
    #im_cropped.show()
    #return classify(im_cropped, threshold)
    return True
