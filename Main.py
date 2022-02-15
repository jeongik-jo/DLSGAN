import HyperParameters as hp
import Train
import time
import Models
import Dataset
import Evaluate
import datetime
import numpy as np
import os
import tensorflow as tf


def train_gan():
    generator = Models.Generator()
    discriminator = Models.Discriminator()

    if hp.load_model:
        generator.load(), discriminator.load()

    train_dataset = Dataset.load_train_dataset()
    test_dataset = Dataset.load_test_dataset()

    fids = []
    precisions = []
    recalls = []

    real_psnrs = []
    real_ssims = []
    fake_psnrs = []
    fake_ssims = []

    enc_losses = []

    latent_entropys = []

    for epoch in range(hp.epochs):
        print(datetime.datetime.now())
        print('epoch', epoch)
        start = time.time()
        enc_loss, var_vectors = Train.train_gan(generator.model, discriminator.model,
                                                             train_dataset, generator.var_vectors)

        hp.generator_optimizer.lr = hp.generator_optimizer.lr * hp.lr_decay_rate
        hp.discriminator_optimizer.lr = hp.discriminator_optimizer.lr * hp.lr_decay_rate

        if hp.is_dls_gan:
            generator.var_vectors = var_vectors
            latent_scale_vector = tf.sqrt(tf.reduce_mean(var_vectors, axis=0, keepdims=True))
            latent_scale_vector = tf.sqrt(tf.cast(hp.latent_vector_dim, dtype='float32')) * latent_scale_vector / tf.norm(latent_scale_vector, axis=-1, keepdims=True)
        else:
            latent_scale_vector = tf.ones([1, hp.latent_vector_dim])

        latent_entropy = hp.latent_entropy_func(latent_scale_vector)
        print('latent entropy:', latent_entropy.numpy())
        print('enc loss:', enc_loss.numpy(), '\n')

        print('saving...')
        generator.save()
        discriminator.save()
        generator.save_images(test_dataset, discriminator.model, latent_scale_vector, epoch)
        print('saved')
        print('time: ', time.time() - start, '\n')
        if hp.evaluate_model and (epoch + 1) % hp.epoch_per_evaluate == 0:
            print('evaluating...', '\n')
            start = time.time()

            latent_entropys.append(latent_entropy)
            enc_losses.append(enc_loss)

            fid, precision, recall = Evaluate.get_fid_pr(generator.model, test_dataset, latent_scale_vector)
            fids.append(fid)
            precisions.append(precision)
            recalls.append(recall)
            print('fid:', fid.numpy())
            print('precision:', precision.numpy())
            print('recall:', recall.numpy())

            fake_psnr, fake_ssim = Evaluate.evaluate_fake_rec(generator.model, discriminator.model, test_dataset, latent_scale_vector)
            fake_psnrs.append(fake_psnr), fake_ssims.append(fake_ssim)
            print('\nfake psnr:', fake_psnr.numpy(), '\nfake ssim:', fake_ssim.numpy())

            real_psnr, real_ssim = Evaluate.evaluate_real_rec(generator.model, discriminator.model, test_dataset, latent_scale_vector)
            real_psnrs.append(real_psnr), real_ssims.append(real_ssim)
            print('\nreal psnr:', real_psnr.numpy(), '\nreal ssim:', real_ssim.numpy())

            print('\n', 'evaluated')
            print('time: ', time.time() - start, '\n')
            if not os.path.exists('./results'):
                os.makedirs('./results')

            np.savetxt('./results/fids.txt', np.array(fids), fmt='%f')
            np.savetxt('./results/precisions.txt', np.array(precisions), fmt='%f')
            np.savetxt('./results/recalls.txt', np.array(recalls), fmt='%f')

            np.savetxt('./results/fake_psnrs.txt', np.array(fake_psnrs), fmt='%f')
            np.savetxt('./results/fake_ssims.txt', np.array(fake_ssims), fmt='%f')
            np.savetxt('./results/real_psnrs.txt', np.array(real_psnrs), fmt='%f')
            np.savetxt('./results/real_ssims.txt', np.array(real_ssims), fmt='%f')

            np.savetxt('./results/enc_losses.txt', np.array(enc_losses), fmt='%f')
            np.savetxt('./results/latent_entropys.txt', np.array(latent_entropys), fmt='%f')


def train_encoder():
    generator = Models.Generator()
    discriminator = Models.Discriminator()

    generator.load()

    train_dataset = Dataset.load_train_dataset()
    test_dataset = Dataset.load_test_dataset()

    real_psnrs = []
    real_ssims = []
    fake_psnrs = []
    fake_ssims = []
    enc_losses = []

    for epoch in range(hp.epochs):
        print(datetime.datetime.now())
        print('epoch', epoch)
        start = time.time()
        enc_loss = Train.train_encoder(generator.model, discriminator.model, train_dataset)
        hp.discriminator_optimizer.lr = hp.discriminator_optimizer.lr * hp.lr_decay_rate

        latent_scale_vector = tf.ones([1, hp.latent_vector_dim])
        print('enc loss:', enc_loss.numpy(), '\n')

        print('saving...')
        generator.save()
        discriminator.save()
        generator.save_images(test_dataset, discriminator.model, latent_scale_vector, epoch)
        print('saved')
        print('time: ', time.time() - start, '\n')
        if hp.evaluate_model and (epoch + 1) % hp.epoch_per_evaluate == 0:
            start = time.time()
            enc_losses.append(enc_loss)

            print('evaluating...', '\n')
            fake_psnr, fake_ssim = Evaluate.evaluate_fake_rec(generator.model, discriminator.model,
                                                              test_dataset, latent_scale_vector)
            fake_psnrs.append(fake_psnr), fake_ssims.append(fake_ssim)
            print('\nfake psnr:', fake_psnr.numpy(), '\nfake ssim:', fake_ssim.numpy())

            real_psnr, real_ssim = Evaluate.evaluate_real_rec(generator.model, discriminator.model,
                                                              test_dataset, latent_scale_vector)
            real_psnrs.append(real_psnr), real_ssims.append(real_ssim)
            print('\nreal psnr:', real_psnr.numpy(), '\nreal ssim:', real_ssim.numpy())

            print('\n', 'evaluated')
            print('time: ', time.time() - start, '\n')
            if not os.path.exists('./results'):
                os.makedirs('./results')

            np.savetxt('./results/fake_psnrs.txt', np.array(fake_psnrs), fmt='%f')
            np.savetxt('./results/fake_ssims.txt', np.array(fake_ssims), fmt='%f')
            np.savetxt('./results/real_psnrs.txt', np.array(real_psnrs), fmt='%f')
            np.savetxt('./results/real_ssims.txt', np.array(real_ssims), fmt='%f')

            np.savetxt('./results/enc_losses.txt', np.array(enc_losses), fmt='%f')


train_gan()

