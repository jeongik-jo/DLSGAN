import tensorflow as tf
import HyperParameters as hp
import os


def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example = tf.io.decode_jpeg(example["image"], channels=3)

    return example


def _load_dataset(path):
    dataset = tf.data.TFRecordDataset([path + '/' + file for file in os.listdir(path)],
                                      num_parallel_reads=tf.data.AUTOTUNE)
    return dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)


def load_train_dataset():
    dataset = _load_dataset('dataset/train').shuffle(10000)

    if hp.train_data_size != -1:
        dataset = dataset.take(hp.train_data_size)

    return dataset.batch(hp.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def load_test_dataset():
    dataset = _load_dataset('dataset/test')

    if hp.shuffle_test_dataset:
        dataset = dataset.shuffle(10000)

    if hp.test_data_size != -1:
        dataset = dataset.take(hp.test_data_size)
    return dataset.batch(hp.fid_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


@tf.function
def resize_and_normalize(images):
    images = tf.image.resize(images=images, size=[hp.image_resolution, hp.image_resolution])
    images = tf.cast(images, dtype='float32') / 127.5 - 1.0

    return images
