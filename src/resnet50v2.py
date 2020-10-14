# run as script
# Feature extraction w/ data augmentation.
# Extend our classifier on top of the ResNet50V2, then run the whole thing

from keras.applications import ResNet50V2
from preprocess import *
from model_utils import *
from data_utils import *


def run_model():
    # ResNet50V2
    conv_base = ResNet50V2(weights='imagenet',
                           include_top=False,
                           input_shape=(150, 150, 3))

    model = build_model(conv_base)
    freeze_conv_base(model, conv_base)
    model = compile_model(model)

    train_generator = train_generator(TRAIN_BY_CLASS)
    validation_generator = validation_generator(VALIDATION_BY_CLASS)

    model, history = fit_model(model, train_generator, validation_generator)
    save_model(model, "resnet50v2_model1")


if __name__ == "__main__":
    run_model()