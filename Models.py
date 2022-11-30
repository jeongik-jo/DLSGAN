import tensorflow as tf
from tensorflow import keras as kr
import Layers
import os
import HyperParameters as hp
import numpy as np


class Generator(object):
    def build_model(self):
        latent_vector = kr.Input([hp.latent_vector_dim])
        return kr.Model(latent_vector, Layers.Generator()(latent_vector))

    def __init__(self):
        self.model = self.build_model()
        self.latent_var_trace = tf.Variable(tf.ones([hp.latent_vector_dim]))
        self.save_latent_vectors = [hp.latent_dist_func(hp.save_image_size) for _ in range(hp.save_image_size)]

    def save_images(self, discriminator: kr.Model, test_dataset: tf.data.Dataset, epoch):
        latent_scale_vector = tf.sqrt(tf.cast(hp.latent_vector_dim, 'float32') * self.latent_var_trace / tf.reduce_sum(self.latent_var_trace))
        if not os.path.exists('results/samples'):
            os.makedirs('results/samples')
        # --------------------------------------------------------------------------------------------------------------
        def save_fake_images():
            path = 'results/samples/fake_images'
            if not os.path.exists(path):
                os.makedirs(path)

            images = []
            for i in range(hp.save_image_size):
                fake_images = self.model(self.save_latent_vectors[i] * latent_scale_vector[tf.newaxis])
                images.append(np.hstack(fake_images))

            kr.preprocessing.image.save_img(path=path + '/fake_%d.png' % epoch,
                                            x=tf.clip_by_value(np.vstack(images), clip_value_min=-1, clip_value_max=1))
        save_fake_images()
        #--------------------------------------------------------------------------------------------------------------
        def save_rec_images(is_real):
            if is_real:
                path = 'results/samples/real_rec_images'
            else:
                path = 'results/samples/fake_rec_images'
            if not os.path.exists(path):
                os.makedirs(path)

            images = []
            for real_images in test_dataset.take(hp.save_image_size // 2):
                if is_real:
                    input_images = real_images[:hp.save_image_size]
                else:
                    input_images = self.model(hp.latent_dist_func(hp.save_image_size) * latent_scale_vector[tf.newaxis])
                _, rec_latent_vectors = discriminator(input_images)
                rec_images = self.model(rec_latent_vectors * latent_scale_vector[tf.newaxis])

                images.append(np.vstack(input_images))
                images.append(np.vstack(rec_images))
                images.append(tf.ones([np.vstack(input_images).shape[0], 5, 3]))

            images = tf.clip_by_value(np.hstack(images), clip_value_min=-1, clip_value_max=1)
            if is_real:
                kr.preprocessing.image.save_img(path=path + '/real_rec_%d.png' % epoch, x=images)
            else:
                kr.preprocessing.image.save_img(path=path + '/fake_rec_%d.png' % epoch, x=images)

        save_rec_images(True)
        save_rec_images(False)
        # --------------------------------------------------------------------------------------------------------------
        def save_interpolation_images():
            path = 'results/samples/latent_interpolation'
            if not os.path.exists(path):
                os.makedirs(path)

            indexes = tf.argsort(latent_scale_vector, axis=-1, direction='DESCENDING')
            interpolation_values = tf.linspace(-hp.latent_interpolation_value, hp.latent_interpolation_value,
                                               hp.save_image_size)[:, tf.newaxis]
            latent_vectors = hp.latent_dist_func(hp.save_image_size)
            for i in range(hp.save_image_size):
                images = []
                mask = tf.one_hot(indexes[i], axis=-1, depth=hp.latent_vector_dim)[tf.newaxis]
                for j in range(hp.save_image_size):
                    interpolation_latent_vectors = latent_vectors[j][tf.newaxis] * (1 - mask) + interpolation_values * mask
                    images.append(np.hstack(
                        self.model(interpolation_latent_vectors * latent_scale_vector[tf.newaxis])))

                kr.preprocessing.image.save_img(
                    path=path + '/latent_interpolation_%d_%d.png' % (epoch, i),
                    x=tf.clip_by_value(np.vstack(images), clip_value_min=-1, clip_value_max=1))

        save_interpolation_images()
        # --------------------------------------------------------------------------------------------------------------
    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save_weights('models/generator.h5')
        np.save('models/latent_var_trace.npy', self.latent_var_trace)

    def load(self):
        self.model.load_weights('models/generator.h5')
        self.latent_var_trace.assign(np.load('models/latent_var_trace.npy'))

    def to_ema(self):
        self.train_weights = [tf.constant(weight) for weight in self.model.trainable_variables]
        for weight in self.model.trainable_variables:
            weight.assign(hp.generator_ema.average(weight))

    def to_train(self):
        for ema_weight, train_weight in zip(self.model.trainable_variables, self.train_weights):
            ema_weight.assign(train_weight)


class Discriminator(object):
    def build_model(self):
        input_image = kr.Input([hp.image_resolution, hp.image_resolution, 3])
        return kr.Model(input_image, Layers.Discriminator()(input_image))

    def __init__(self):
        self.model = self.build_model()

    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save_weights('models/discriminator.h5')

    def load(self):
        self.model.load_weights('models/discriminator.h5')

    def to_ema(self):
        self.train_weights = [tf.constant(weight) for weight in self.model.trainable_variables]
        for weight in self.model.trainable_variables:
            weight.assign(hp.discriminator_ema.average(weight))

    def to_train(self):
        for ema_weight, train_weight in zip(self.model.trainable_variables, self.train_weights):
            ema_weight.assign(train_weight)
