# NOT a script
# model building utilities
# loss & accuracy plotting utilities

import matplotlib.pyplot as plt

from keras import layers, models, optimizers
from keras.callbacks import TensorBoard
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
                  metrics=['acc'])
    return model


def fit_model(model, train_generator, validation_generator):
    callbacks = [TensorBoard("logs")]
    history = model.fit(train_generator,
                        # steps_per_epoch=100,
                        epochs=10,
                        validation_data=validation_generator,
                        # validation_steps=50,
                        callbacks=callbacks)
    return model, history


def save_model(model, name):
    model.save(f'{name}.h5')


def plot_acc_loss(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()