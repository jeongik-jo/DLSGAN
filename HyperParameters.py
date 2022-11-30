import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow import keras as kr

generator_optimizer = kr.optimizers.Adam(learning_rate=0.003, beta_1=0.0, beta_2=0.99)
generator_ema = tf.train.ExponentialMovingAverage(decay=0.999)
discriminator_optimizer = kr.optimizers.Adam(learning_rate=0.003, beta_1=0.0, beta_2=0.99)
discriminator_ema = tf.train.ExponentialMovingAverage(decay=0.999)

image_resolution = 256
latent_vector_dim = 1024

reg_weight = 3.0
d_enc_weight = 1.0
g_enc_weight = 1.0
latent_var_decay_rate = 0.999
batch_size = 16
save_image_size = 8

train_only_encoder = False
train_data_size = -1
test_data_size = -1
shuffle_test_dataset = False
epochs = 100

load_model = False

evaluate_model = True
fid_batch_size = batch_size
epoch_per_evaluate = 1


def latent_dist_func(batch_size):
    return tf.random.normal([batch_size, latent_vector_dim])
def latent_entropy_func(latent_scale_vector):
    return tf.reduce_sum(tf.math.log(latent_scale_vector * tf.sqrt(2.0 * 3.141592 * tf.exp(1.0))))
latent_interpolation_value = 2.0
