import tensorflow as tf
import tensorflow.keras as kr
import Dataset
import Layers
import os
import HyperParameters as hp
import numpy as np


class Generator(object):
    def build_model(self):
        latent_vector = kr.Input([hp.latent_vector_dim])
        fake_image = Layers.Generator()(latent_vector)
        return kr.Model(latent_vector, fake_image)

    def __init__(self):
        self.model = self.build_model()
        self.var_vectors = tf.ones([hp.var_vector_size, hp.latent_vector_dim])
        self.save_latent_vectors = hp.latent_dist_func([hp.save_image_size, hp.save_image_size, hp.latent_vector_dim])

    def save_images(self, test_dataset: tf.data.Dataset, discriminator: kr.Model, latent_scale_vector, epoch):
        def save_fake_images():
            if not os.path.exists('./results/fake_images'):
                os.makedirs('./results/fake_images')
            images = []
            for i in range(hp.save_image_size):
                fake_images = self.model(self.save_latent_vectors[i] * latent_scale_vector, training=False)
                images.append(np.hstack(fake_images))

            kr.preprocessing.image.save_img(path='./results/fake_images/fake_%d.png' % epoch,
                                            x=tf.clip_by_value(np.vstack(images), clip_value_min=-1, clip_value_max=1))
        save_fake_images()
        # --------------------------------------------------------------------------------------------------------------
        def save_real_rec_images():
            if not os.path.exists('./results/real_rec_images'):
                os.makedirs('./results/real_rec_images')
            images = []
            for data in test_dataset.take(hp.save_image_size // 2):
                real_images = Dataset.resize_and_normalize(data[:hp.save_image_size])
                _, rec_latent_vectors = discriminator(real_images, training=False)
                rec_images = self.model(rec_latent_vectors * latent_scale_vector, training=False)

                images.append(np.vstack(real_images))
                images.append(np.vstack(rec_images))
                images.append(tf.ones([np.vstack(real_images).shape[0], 5, 3]))

            kr.preprocessing.image.save_img(path='./results/real_rec_images/real_rec_%d.png' % epoch,
                                            x=tf.clip_by_value(np.hstack(images), clip_value_min=-1, clip_value_max=1))
        save_real_rec_images()
        # --------------------------------------------------------------------------------------------------------------
        def save_fake_rec_images():
            if not os.path.exists('./results/fake_rec_images'):
                os.makedirs('./results/fake_rec_images')
            images = []
            for _ in test_dataset.take(hp.save_image_size // 2):
                latent_vectors = hp.latent_dist_func([hp.save_image_size, hp.latent_vector_dim])
                generated_images = self.model(latent_vectors * latent_scale_vector, training=False)
                _, rec_latent_vectors = discriminator(generated_images, training=False)
                recovered_images = self.model(rec_latent_vectors * latent_scale_vector, training=False)

                images.append(np.vstack(generated_images))
                images.append(np.vstack(recovered_images))
                images.append(tf.ones([np.vstack(generated_images).shape[0], 5, 3]))

            kr.preprocessing.image.save_img(path='./results/fake_rec_images/fake_rec_%d.png' % epoch,
                                            x=tf.clip_by_value(np.hstack(images), clip_value_min=-1, clip_value_max=1))
        save_fake_rec_images()
        # --------------------------------------------------------------------------------------------------------------
        def save_latent_noised_images():
            if not os.path.exists('./results/latent_noised_images'):
                os.makedirs('./results/latent_noised_images')
            images = []
            for data in test_dataset.take(1):
                real_images = Dataset.resize_and_normalize(data[:hp.save_image_size])
                _, rec_latent_vectors = discriminator(real_images, training=False)
                reconstructed_images = self.model(rec_latent_vectors * latent_scale_vector, training=False)
                noised_vectors_sets = hp.latent_add_noises(rec_latent_vectors, hp.save_image_size)

                for i in range(hp.save_image_size):
                    images.append(np.hstack([
                        real_images[i],
                        reconstructed_images[i],
                        tf.ones([hp.image_resolution, 5, 3]),
                        np.hstack(self.model(noised_vectors_sets[i] * latent_scale_vector, training=False))]))

            kr.preprocessing.image.save_img(path='./results/latent_noised_images/latent_noised_%d.png' % epoch,
                                            x=tf.clip_by_value(np.vstack(images), clip_value_min=-1, clip_value_max=1))
        save_latent_noised_images()
        #---------------------------------------------------------------------------------------------------------------
        if hp.is_dls_gan:
            def save_interpolation_images():
                if not os.path.exists('./results/latent_interpolation'):
                    os.makedirs('./results/latent_interpolation')

                indexes = tf.argsort(latent_scale_vector, axis=-1, direction='DESCENDING')
                interpolation_values = tf.linspace(-hp.latent_interpolation_value, hp.latent_interpolation_value, hp.save_image_size)[:, tf.newaxis]
                latent_vectors = hp.latent_dist_func([hp.save_image_size, hp.latent_vector_dim])
                for i in range(hp.save_image_size):
                    latent_interpolation_images = []
                    mask = tf.one_hot(indexes[:, i], axis=-1, depth=hp.latent_vector_dim)
                    for j in range(hp.save_image_size):
                        interpolation_latent_vectors = latent_vectors[j][tf.newaxis] * (1 - mask) + interpolation_values * mask
                        latent_interpolation_images.append(np.hstack(
                            self.model(interpolation_latent_vectors * latent_scale_vector, training=False)))

                    kr.preprocessing.image.save_img(
                        path='./results/latent_interpolation/latent_interpolation_%d_%d.png' % (epoch, i),
                        x=tf.clip_by_value(np.vstack(latent_interpolation_images), clip_value_min=-1, clip_value_max=1))

            save_interpolation_images()

    def save(self):
        if not os.path.exists('./models'):
            os.makedirs('./models')
        self.model.save_weights('./models/generator.h5')
        np.save('./models/var_vectors.npy', self.var_vectors)

    def load(self):
        self.model.load_weights('./models/generator.h5')
        self.var_vectors = np.load('./models/var_vectors.npy')


class Discriminator(object):
    def build_model(self):
        input_image = kr.Input([hp.image_resolution, hp.image_resolution, 3])
        adv_value, feature_vector = Layers.Discriminator()(input_image)
        return kr.Model(input_image, [adv_value, feature_vector])

    def __init__(self):
        self.model = self.build_model()

    def save(self):
        if not os.path.exists('./models'):
            os.makedirs('./models')
        self.model.save_weights('./models/discriminator.h5')

    def load(self):
        self.model.load_weights('./models/discriminator.h5')
