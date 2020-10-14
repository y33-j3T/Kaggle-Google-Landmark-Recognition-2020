# NOT a script
# image generators for models

from keras.preprocessing.image import ImageDataGenerator


def train_datagen():
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    return train_datagen


def test_datagen():
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    return test_datagen


def train_generator(train_by_class_dir):
    train_generator = train_datagen.flow_from_directory(
        train_by_class_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')
    return train_generator


def validation_generator(validation_by_class_dir):
    validation_generator = test_datagen.flow_from_directory(
        validation_by_class_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')
    return validation_generator