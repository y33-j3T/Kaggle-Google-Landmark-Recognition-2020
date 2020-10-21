# run as script
# Feature extraction w/ data augmentation.
# Extend our classifier on top of the ResNet50V2, then run the whole thing

import os
import re
import shutil
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.callbacks import TensorBoard
from sklearn.metrics import classification_report
from glob import glob

from model_utils import *

# base directory
BASE_DIR = '..'

# data directory
DATA_DIR = os.path.join(BASE_DIR, 'data')

# raw, interim & processed data directory
RAW_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# raw data
TRAIN = os.path.join(RAW_DIR, 'train')
TEST = os.path.join(RAW_DIR, 'test')
DF_TRAIN = os.path.join(TRAIN, 'train.csv')
SAMPLE_SUBMISSION = os.path.join(RAW_DIR, 'sample_submission.csv')

# interim data
TRAIN_BY_CLASS = os.path.join(INTERIM_DIR, 'train')
VALIDATION_BY_CLASS = os.path.join(INTERIM_DIR, 'validation')
TEST_BY_CLASS = os.path.join(INTERIM_DIR, 'test')

# ResNet50V2
conv_base = ResNet50(weights='imagenet', 
                  include_top=False,
                  input_shape=(150, 150, 3))

# Datagen
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Generator
train_generator = train_datagen.flow_from_directory(
    TRAIN_BY_CLASS,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    VALIDATION_BY_CLASS,
    target_size=(150, 150),
    batch_size=32,
    shuffle=False,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    TEST_BY_CLASS,
    target_size=(150, 150),
    batch_size=1,
    shuffle=False,
    class_mode='categorical')


def main():

    num_class = len(glob(os.path.join(TRAIN_BY_CLASS, '*')))
    model = build_model(conv_base, num_class)
    freeze_conv_base(model, conv_base)
    model = compile_model(model)

    # train_generator = train_generator(TRAIN_BY_CLASS)
    # validation_generator = validation_generator(VALIDATION_BY_CLASS)

    model, history = fit_model(model, train_generator, validation_generator)
    save_model(model, "211020_resnet50v2_1")


if __name__ == "__main__":
    main()