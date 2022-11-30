import HyperParameters as hp
import Train
import time
import Models
import Dataset
import Evaluate
import datetime
import numpy as np
import os
import matplotlib.pyplot as plt


def train_gan():
    discriminator = Models.Discriminator()
    generator = Models.Generator()

    if hp.train_only_encoder:
        generator.load()

    if hp.load_model:
        discriminator.load(), generator.load()

    train_dataset = Dataset.load_train_dataset()
    test_dataset = Dataset.load_test_dataset()

    results = {}
    for epoch in range(hp.epochs):
        print(datetime.datetime.now())
        print('epoch', epoch)
        start = time.time()

        train_results = Train.train(discriminator.model, generator.model, generator.latent_var_trace, train_dataset)
        print('saving...')
        discriminator.to_ema()
        generator.to_ema()
        discriminator.save()
        generator.save()
        generator.save_images(discriminator.model, test_dataset, epoch)
        print('saved')
        print('time: ', time.time() - start, '\n')

        if hp.evaluate_model and (epoch + 1) % hp.epoch_per_evaluate == 0:
            print('evaluating...')
            start = time.time()
            evaluate_results = Evaluate.evaluate(discriminator.model, generator.model,
                                                 generator.latent_var_trace, test_dataset)
            for key in train_results:
                try:
                    results[key].append(train_results[key])
                except KeyError:
                    results[key] = [train_results[key]]
            for key in evaluate_results:
                try:
                    results[key].append(evaluate_results[key])
                except KeyError:
                    results[key] = [evaluate_results[key]]

            print('evaluated')
            print('time: ', time.time() - start, '\n')
            if not os.path.exists('results/figures'):
                os.makedirs('results/figures')
            for key in results:
                np.savetxt('results/figures/%s.txt' % key, results[key], fmt='%f')
                plt.title(key)
                plt.xlabel('Epochs')
                plt.ylabel(key)
                plt.plot([(i + 1) * hp.epoch_per_evaluate for i in range(len(results[key]))], results[key])
                plt.savefig('results/figures/%s.png' % key)
                plt.clf()

        discriminator.to_train()
        generator.to_train()


train_gan()

