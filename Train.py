import tensorflow as tf
from tensorflow import keras as kr
import HyperParameters as hp


@tf.function
def _gan_train_step(discriminator: kr.Model, generator: kr.Model, latent_var_trace: tf.Variable, real_images: tf.Tensor):
    with tf.GradientTape(persistent=True) as tape:
        batch_size = real_images.shape[0]
        latent_vector_dim = tf.cast(hp.latent_vector_dim, 'float32')
        latent_scale_vector = tf.sqrt(latent_vector_dim * latent_var_trace / tf.reduce_sum(latent_var_trace))
        latent_vectors = hp.latent_dist_func(batch_size)
        fake_images = generator(latent_vectors * latent_scale_vector[tf.newaxis])

        fake_adv_values, rec_latent_vectors = discriminator(fake_images)
        enc_losses = tf.reduce_mean(tf.square((latent_vectors - rec_latent_vectors) * latent_scale_vector[tf.newaxis]), axis=-1)

        with tf.GradientTape() as reg_tape:
            reg_tape.watch(real_images)
            real_adv_values, _ = discriminator(real_images)

        real_grads = reg_tape.gradient(real_adv_values, real_images)
        reg_losses = tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3])

        dis_adv_losses = tf.nn.softplus(-real_adv_values) + tf.nn.softplus(fake_adv_values)
        gen_adv_losses = tf.nn.softplus(-fake_adv_values)
        dis_losses = dis_adv_losses + hp.d_enc_weight * enc_losses + hp.reg_weight * reg_losses
        gen_losses = gen_adv_losses + hp.g_enc_weight * enc_losses

        dis_loss = tf.reduce_mean(dis_losses)
        gen_loss = tf.reduce_mean(gen_losses)

    hp.discriminator_optimizer.apply_gradients(
        zip(tape.gradient(dis_loss, discriminator.trainable_variables),
            discriminator.trainable_variables)
    )
    hp.generator_optimizer.apply_gradients(
        zip(tape.gradient(gen_loss, generator.trainable_variables),
            generator.trainable_variables)
    )

    del tape

    hp.discriminator_ema.apply(discriminator.trainable_variables)
    hp.generator_ema.apply(generator.trainable_variables)
    latent_var_trace.assign(latent_var_trace * hp.latent_var_decay_rate +
                            tf.reduce_mean(tf.square(rec_latent_vectors), axis=0) * (1.0 - hp.latent_var_decay_rate))

    results = {
        'real_adv_values': real_adv_values, 'fake_adv_values': fake_adv_values,
        'reg_losses': reg_losses, 'enc_losses': enc_losses,
    }
    return results


@tf.function
def _encoder_train_step(discriminator: kr.Model, generator: kr.Model, latent_var_trace: tf.Variable, real_images: tf.Tensor):
    with tf.GradientTape(persistent=True) as tape:
        batch_size = real_images.shape[0]
        latent_vector_dim = tf.cast(hp.latent_vector_dim, 'float32')
        latent_scale_vector = tf.sqrt(latent_vector_dim * latent_var_trace / tf.reduce_sum(latent_var_trace))
        latent_vectors = hp.latent_dist_func(batch_size)
        fake_images = generator(latent_vectors * latent_scale_vector[tf.newaxis])

        _, rec_latent_vectors = discriminator(fake_images)
        enc_losses = tf.reduce_mean(tf.square((latent_vectors - rec_latent_vectors) * latent_scale_vector[tf.newaxis]), axis=-1)

        dis_losses = hp.d_enc_weight * enc_losses
        dis_loss = tf.reduce_mean(dis_losses)

    hp.discriminator_optimizer.apply_gradients(
        zip(tape.gradient(dis_loss, discriminator.trainable_variables),
            discriminator.trainable_variables)
    )

    del tape

    hp.discriminator_ema.apply(discriminator.trainable_variables)
    hp.generator_ema.apply(generator.trainable_variables)
    latent_var_trace.assign(latent_var_trace * hp.latent_var_decay_rate +
                            tf.reduce_mean(tf.square(rec_latent_vectors), axis=0) * (1.0 - hp.latent_var_decay_rate))

    results = {'enc_losses': enc_losses}
    return results


def train(discriminator: kr.Model, generator: kr.Model, latent_var_trace: tf.Variable, dataset):
    results = {}
    for data in dataset:
        if hp.train_only_encoder:
            batch_results = _encoder_train_step(discriminator, generator, latent_var_trace, data)
        else:
            batch_results = _gan_train_step(discriminator, generator, latent_var_trace, data)
        for key in batch_results:
            try:
                results[key].append(batch_results[key])
            except KeyError:
                results[key] = [batch_results[key]]

    temp_results = {}
    for key in results:
        mean, variance = tf.nn.moments(tf.concat(results[key], axis=0), axes=0)
        temp_results[key + '_mean'] = mean
        temp_results[key + '_variance'] = variance
    results = temp_results

    for key in results:
        print('%-30s:' % key, '%13.6f' % results[key].numpy())

    return results
