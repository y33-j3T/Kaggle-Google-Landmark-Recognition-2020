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
from keras.applications import ResNet50V2
from keras.callbacks import TensorBoard
from sklearn.metrics import classification_report
from glob import glob

from model_utils import *
from data_utils import *
from path import *


################
# Define Stuff #
################
# base convolutional nn
CONV_BASE = ResNet50V2(weights='imagenet', 
                  include_top=False,
                  input_shape=(150, 150, 3))

# no. of epochs
EPOCH = 30

# parameters to name models & logs
DATE = "221020"
MODEL = "resnet50v2"
VERSION = "2"

# directory to store models & logs
callback_dir = os.path.join(CALLBACK_DIR, DATE, MODEL, VERSION)
models_dir = os.path.join(MODELS_DIR, DATE, MODEL, VERSION)

# no. of output classes
num_class = len(glob(os.path.join(TRAIN_BY_CLASS, '*')))


def main():
    try:
        os.makedirs(callback_dir)
        os.makedirs(models_dir)
    except OSError as e:
        print(e)

    model = build_model(CONV_BASE, num_class)
    freeze_conv_base(model, CONV_BASE)
    model = compile_model(model)

    model, history = fit_model(model, TRAIN_GENERATOR, VALIDATION_GENERATOR, EPOCH, callback_dir, models_dir)
    save_model(model, models_dir)


if __name__ == "__main__":
    main()