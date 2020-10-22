##################
# Model Building #
##################

import matplotlib.pyplot as plt
import os

from keras import layers, models, optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.metrics import Accuracy, CategoricalAccuracy, Precision
from glob import glob


def build_model(conv_base, num_class):
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_class, activation='softmax'))
    return model


def freeze_conv_base(model, conv_base):
    print(
        'This is the number of trainable weights '
        'before freezing the conv base:', len(model.trainable_weights))

    # Freeze the convolutional base to prevent their weights from being updated during training
    # We only want to train the newly added layers
    conv_base.trainable = False

    print(
        'This is the number of trainable weights '
        'after freezing the conv base:', len(model.trainable_weights))


def compile_model(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=2e-5),
                  metrics=[
                        'acc', 
                        CategoricalAccuracy(name="categorical_accuracy", dtype=None),
                        Precision()
                    ]
    )
    return model


def fit_model(model, train_generator, validation_generator, epochs, callback_dir, models_dir):
    callbacks = [
        TensorBoard(callback_dir),
        ModelCheckpoint(filepath=os.path.join(models_dir, 'model.{epoch:02d}-{val_loss:.2f}.h5'))
    ]
    history = model.fit(train_generator,
                        # steps_per_epoch=100,
                        epochs=epochs,
                        validation_data=validation_generator,
                        # validation_steps=50,
                        callbacks=callbacks)
    return model, history


def save_model(model, dir):
    # if not os.path.exists(dir):
    #     try:
    #         os.makedirs(dir)
    #     except OSError as e:
    #         print(e)

    name = 1
    name = os.path.join(dir, f'{name}.h5')
    model.save(name)
