from tensorflow import keras as kr
import tensorflow as tf
import tensorflow_probability as tfp
import HyperParameters as hp
from scipy.linalg import sqrtm
import numpy as np

inception_model = tf.keras.applications.InceptionV3(weights='imagenet', pooling='avg', include_top=False)


@tf.function
def _get_batch_results(discriminator: kr.Model, generator: kr.Model, latent_scale_vector, real_images):
    batch_size = real_images.shape[0]
    latent_vectors = hp.latent_dist_func(batch_size)
    fake_images = tf.clip_by_value(generator(latent_vectors * latent_scale_vector[tf.newaxis]), clip_value_min=-1, clip_value_max=1)

    real_rec_images = tf.clip_by_value(generator(discriminator(real_images)[1] * latent_scale_vector[tf.newaxis]), clip_value_min=-1, clip_value_max=1)
    fake_rec_images = tf.clip_by_value(generator(discriminator(fake_images)[1] * latent_scale_vector[tf.newaxis]), clip_value_min=-1, clip_value_max=1)

    real_features = inception_model(tf.image.resize(real_images, [299, 299]))
    fake_features = inception_model(tf.image.resize(fake_images, [299, 299]))

    real_psnrs = tf.image.psnr(real_images, real_rec_images, max_val=2.0)
    real_ssims = tf.image.ssim(real_images, real_rec_images, max_val=2.0)
    fake_psnrs = tf.image.psnr(fake_images, fake_rec_images, max_val=2.0)
    fake_ssims = tf.image.ssim(fake_images, fake_rec_images, max_val=2.0)

    return {'real_psnrs': real_psnrs, 'real_ssims': real_ssims, 'fake_psnrs': fake_psnrs, 'fake_ssims': fake_ssims,
            'real_features': real_features, 'fake_features': fake_features}


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


def evaluate(discriminator: kr.Model, generator: kr.Model, latent_var_trace: tf.Tensor, test_dataset: tf.data.Dataset):
    latent_scale_vector = tf.sqrt(tf.cast(hp.latent_vector_dim, 'float32') * latent_var_trace / tf.reduce_sum(latent_var_trace))
    results = {}
    for real_images in test_dataset:
        batch_results = _get_batch_results(discriminator, generator, latent_scale_vector, real_images)
        for key in batch_results:
            try:
                results[key].append(batch_results[key])
            except KeyError:
                results[key] = [batch_results[key]]

    real_psnr = tf.reduce_mean(results['real_psnrs'])
    real_ssim = tf.reduce_mean(results['real_ssims'])
    fake_psnr = tf.reduce_mean(results['fake_psnrs'])
    fake_ssim = tf.reduce_mean(results['fake_ssims'])

    real_features = tf.concat(results['real_features'], axis=0)
    fake_features = tf.concat(results['fake_features'], axis=0)

    fid = _get_fid(real_features, fake_features)
    precision = _get_pr(real_features, fake_features)
    recall = _get_pr(fake_features, real_features)

    latent_entropy = hp.latent_entropy_func(latent_scale_vector)

    results = {'fid': fid, 'precision': precision, 'recall': recall,
               'real_psnr': real_psnr, 'real_ssim': real_ssim, 'fake_psnr': fake_psnr, 'fake_ssim': fake_ssim,
               'latent_entropy': latent_entropy}

    for key in results:
        print('%-20s:' % key, '%13.6f' % results[key].numpy())

    return results






