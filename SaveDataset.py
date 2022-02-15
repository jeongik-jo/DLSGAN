import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

folder_path = r'D:\FFHQ\train'
dataset_name = 'FFHQ_train'
resize_image = True
if resize_image:
    resolution = [512, 512]
sample_paths = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        sample_paths.append(os.path.join(root, file))

sample_per_tfr = 10000
num_tfr = len(sample_paths) // sample_per_tfr
if len(sample_paths) % sample_per_tfr != 0:
    num_tfr += 1


def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(image):
    feature = {
        "image": image_feature(image),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


if not os.path.exists('dataset'):
    os.makedirs('dataset')


for i_tfr in range(num_tfr):
    with tf.io.TFRecordWriter("dataset/" + dataset_name + "_%03d.tfrec" % i_tfr) as writer:
        temp_sample_paths = sample_paths[i_tfr*sample_per_tfr:(i_tfr+1)*sample_per_tfr]
        for sample_path in temp_sample_paths:
            image = tf.io.decode_jpeg(tf.io.read_file(sample_path))
            if resize_image:
                image = tf.cast(tf.round(tf.image.resize(tf.cast(image, 'float32'), resolution)), 'uint8')
            example = create_example(image)
            writer.write(example.SerializeToString())
