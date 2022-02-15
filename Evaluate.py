import tensorflow.keras as kr
import tensorflow as tf
import tensorflow_probability as tfp
import HyperParameters as hp
from scipy.linalg import sqrtm
import numpy as np
import Dataset

inception_model = tf.keras.applications.InceptionV3(weights='imagenet', pooling='avg', include_top=False)


@tf.function
def _get_feature_samples(generator: kr.Model, real_images, latent_scale_vector):
    real_images = Dataset.resize_and_normalize(real_images)

    batch_size = real_images.shape[0]
    latent_vectors = hp.latent_dist_func([batch_size, hp.latent_vector_dim])

    fake_images = tf.clip_by_value(generator(latent_vectors * latent_scale_vector, training=False), clip_value_min=-1, clip_value_max=1)
    fake_images = tf.image.resize(fake_images, [299, 299])
    real_images = tf.image.resize(real_images, [299, 299])

    real_features = inception_model(real_images, training=False)
    fake_features = inception_model(fake_images, training=False)

    return real_features, fake_features


def _get_features(generator: kr.Model, test_dataset: tf.data.Dataset, latent_scale_vector):
    real_features = []
    fake_features = []

    for real_images in test_dataset:
        real_features_batch, fake_features_batch = _get_feature_samples(generator, real_images, latent_scale_vector)
        real_features.append(real_features_batch)
        fake_features.append(fake_features_batch)

    real_features = tf.concat(real_features, axis=0)
    fake_features = tf.concat(fake_features, axis=0)

    return real_features, fake_features


def _pairwise_distances(U, V):
    norm_u = tf.reduce_sum(tf.square(U), 1)
    norm_v = tf.reduce_sum(tf.square(V), 1)

    norm_u = tf.reshape(norm_u, [-1, 1])
    norm_v = tf.reshape(norm_v, [1, -1])

    D = tf.maximum(norm_u - 2 * tf.matmul(U, V, False, True) + norm_v, 0.0)

    return D


def _get_fid(real_features, fake_features):
    real_features_mean = tf.reduce_mean(real_features, axis=0)
    fake_features_mean = tf.reduce_mean(fake_features, axis=0)

    mean_difference = tf.reduce_sum((real_features_mean - fake_features_mean) ** 2)
    real_cov, fake_cov = tfp.stats.covariance(real_features), tfp.stats.covariance(fake_features)
    cov_mean = sqrtm(tf.matmul(real_cov, fake_cov))

    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    cov_difference = tf.linalg.trace(real_cov + fake_cov - 2.0 * cov_mean)

    fid = mean_difference + cov_difference

    return fid


@tf.function
def _get_pr(ref_features, eval_features, nhood_size=3):
    thresholds = -tf.math.top_k(-_pairwise_distances(ref_features, ref_features), k=nhood_size + 1, sorted=True)[0]
    thresholds = thresholds[tf.newaxis, :, -1]

    distance_pairs = _pairwise_distances(eval_features, ref_features)
    return tf.reduce_mean(tf.cast(tf.math.reduce_any(distance_pairs <= thresholds, axis=1), 'float32'))


def get_fid_pr(generator: kr.Model, test_dataset: tf.data.Dataset, latent_scale_vector):
    real_features, fake_features = _get_features(generator, test_dataset, latent_scale_vector)

    fid = _get_fid(real_features, fake_features)
    precision = _get_pr(real_features, fake_features)
    recall = _get_pr(fake_features, real_features)

    return fid, precision, recall


def evaluate_fake_rec(generator: kr.Model, discriminator: kr.Model, test_dataset: tf.data.Dataset, latent_scale_vector):
    average_psnr = []
    average_ssim = []
    for _ in test_dataset:
        latent_vectors = hp.latent_dist_func([hp.batch_size, hp.latent_vector_dim])
        fake_images = tf.clip_by_value(
            generator(latent_vectors * latent_scale_vector, training=False),
            clip_value_min=-1, clip_value_max=1)

        _, rec_latent_vectors = discriminator(fake_images, training=False)
        rec_images = tf.clip_by_value(
            generator(rec_latent_vectors * latent_scale_vector, training=False),
            clip_value_min=-1, clip_value_max=1)

        average_psnr.append(tf.reduce_mean(tf.image.psnr(fake_images, rec_images, max_val=2.0)))
        average_ssim.append(tf.reduce_mean(tf.image.ssim(fake_images, rec_images, max_val=2.0)))

    return tf.reduce_mean(average_psnr), tf.reduce_mean(average_ssim)


def evaluate_real_rec(generator: kr.Model, discriminator: kr.Model, test_dataset: tf.data.Dataset, latent_scale_vector):
    average_psnr = []
    average_ssim = []
    for data in test_dataset:
        real_images = Dataset.resize_and_normalize(data)
        _, rec_latent_vectors = discriminator(real_images, training=False)
        rec_images = tf.clip_by_value(
            generator(rec_latent_vectors * latent_scale_vector, training=False),
            clip_value_min=-1, clip_value_max=1)

        average_psnr.append(tf.reduce_mean(tf.image.psnr(real_images, rec_images, max_val=2.0)))
        average_ssim.append(tf.reduce_mean(tf.image.ssim(real_images, rec_images, max_val=2.0)))

    return tf.reduce_mean(average_psnr), tf.reduce_mean(average_ssim)

