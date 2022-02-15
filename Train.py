import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as hp
import Dataset


@tf.function
def _gan_train_step(generator: kr.Model, discriminator: kr.Model, real_images: tf.Tensor, var_vectors):
    real_images = Dataset.resize_and_normalize(real_images)

    with tf.GradientTape(persistent=True) as tape:
        batch_size = real_images.shape[0]
        latent_vectors = hp.latent_dist_func([batch_size, hp.latent_vector_dim])

        if hp.is_dls_gan:
            latent_scale_vector = tf.sqrt(tf.reduce_mean(var_vectors, axis=0, keepdims=True))
            latent_scale_vector = tf.sqrt(tf.cast(hp.latent_vector_dim, dtype='float32')) * latent_scale_vector / tf.norm(latent_scale_vector, axis=-1, keepdims=True)
            fake_images = generator(latent_vectors * latent_scale_vector, training=True)
            fake_adv_values, rec_latent_vectors = discriminator(fake_images, training=True)
            encoder_losses = tf.reduce_mean(
                tf.square(rec_latent_vectors - latent_vectors) * tf.square(latent_scale_vector), axis=-1)
        else:
            fake_images = generator(latent_vectors, training=True)
            fake_adv_values, rec_latent_vectors = discriminator(fake_images, training=True)
            encoder_losses = tf.reduce_mean(tf.square(rec_latent_vectors - latent_vectors), axis=-1)

        with tf.GradientTape() as r1_tape:
            r1_tape.watch(real_images)
            real_adv_values, _ = discriminator(real_images, training=True)
        r1_gradients = r1_tape.gradient(real_adv_values, real_images)
        r1_regs = tf.reduce_sum(tf.square(r1_gradients), axis=[1, 2, 3])

        discriminator_losses = tf.squeeze(tf.nn.softplus(-real_adv_values) + tf.nn.softplus(fake_adv_values)) \
                               + hp.r1_weight * r1_regs + hp.dis_enc_weight * encoder_losses
        discriminator_loss = tf.reduce_mean(discriminator_losses)
        generator_losses = tf.squeeze(tf.nn.softplus(-fake_adv_values)) \
                           + hp.gen_enc_weight * encoder_losses
        generator_loss = tf.reduce_mean(generator_losses)

    var_vectors = tf.concat([var_vectors[1:], tf.reduce_mean(tf.square(rec_latent_vectors), axis=0, keepdims=True)], axis=0)

    hp.generator_optimizer.apply_gradients(
        zip(tape.gradient(generator_loss, generator.trainable_variables),
            generator.trainable_variables)
    )

    hp.discriminator_optimizer.apply_gradients(
        zip(tape.gradient(discriminator_loss, discriminator.trainable_variables),
            discriminator.trainable_variables)
    )

    del tape

    return tf.reduce_mean(encoder_losses), var_vectors


@tf.function
def _encoder_train_step(generator: kr.Model, discriminator: kr.Model):
    latent_vectors = hp.latent_dist_func([hp.batch_size, hp.latent_vector_dim])
    fake_images = generator(latent_vectors, training=False)

    with tf.GradientTape() as tape:
        tape.watch(fake_images)
        _, rec_latent_vectors = discriminator(fake_images, training=True)
        enc_losses = tf.reduce_mean(tf.square(rec_latent_vectors - latent_vectors), axis=-1)
        enc_loss = tf.reduce_mean(enc_losses * hp.dis_enc_weight)

    hp.discriminator_optimizer.apply_gradients(
        zip(tape.gradient(enc_loss, discriminator.trainable_variables),
            discriminator.trainable_variables)
    )

    return enc_loss


def train_gan(generator: kr.Model, discriminator: kr.Model, dataset, var_vectors):
    enc_losses = []
    for data in dataset:
        enc_loss, var_vectors = _gan_train_step(generator, discriminator, data, var_vectors)
        enc_losses.append(enc_loss)
    mean_enc_loss = tf.reduce_mean(enc_losses)

    return mean_enc_loss, var_vectors


def train_encoder(generator: kr.Model, discriminator: kr.Model, dataset):
    enc_losses = []
    for _ in dataset:
        enc_loss = _encoder_train_step(generator, discriminator)
        enc_losses.append(enc_loss)
    mean_enc_loss = tf.reduce_mean(enc_losses)

    return mean_enc_loss
