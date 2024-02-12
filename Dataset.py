import tensorflow as tf
from tensorflow import keras as kr
import HyperParameters as hp


@tf.function
def _normalize(images):
    images = tf.cast(images, 'float32') / 127.5 - 1.0
    return images


def load_train_dataset():
    if hp.is_ffhq:
        directory = 'dataset/ffhq/train'
    else:
        directory = 'dataset/afhq/train'

    dataset = kr.utils.image_dataset_from_directory(directory=directory, labels=None, batch_size=None,
                                                    image_size=[hp.img_res, hp.img_res], shuffle=True, seed=111)

    if hp.train_data_size != -1:
        dataset = dataset.take(hp.train_data_size)

    return dataset.batch(hp.batch_size, drop_remainder=True).map(_normalize)


def load_test_dataset():
    if hp.is_ffhq:
        directory = 'dataset/ffhq/test'
    else:
        directory = 'dataset/afhq/test'

    dataset = kr.utils.image_dataset_from_directory(directory=directory, labels=None, batch_size=None,
                                                    image_size=[hp.img_res, hp.img_res], shuffle=True, seed=111)

    if hp.test_data_size != -1:
        dataset = dataset.take(hp.test_data_size)

    return dataset.batch(hp.batch_size, drop_remainder=True).map(_normalize)


def load_sample_dataset():
    if hp.is_ffhq:
        directory = 'dataset/ffhq/test'
    else:
        directory = 'dataset/afhq/test'

    dataset = kr.utils.image_dataset_from_directory(directory=directory, labels=None, batch_size=None,
                                                    image_size=[hp.img_res, hp.img_res], shuffle=True, seed=111)
    return dataset.batch(hp.batch_size, drop_remainder=True).map(_normalize)
