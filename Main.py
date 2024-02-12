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
    dis = Models.Discriminator()
    gen = Models.Generator()

    if hp.train_only_enc:
        gen.load()
    elif hp.load_model:
        dis.load(), gen.load()

    train_dataset = Dataset.load_train_dataset()
    test_dataset = Dataset.load_test_dataset()

    results = {}
    for epoch in range(hp.epochs):
        print(datetime.datetime.now())
        print('epoch', epoch)
        start = time.time()

        train_results = Train.train(dis.model, gen.model, train_dataset)
        print('saving...')
        dis.to_ema()
        if not hp.train_only_enc:
            gen.to_ema()
        dis.save()
        gen.save()
        gen.save_imgs(dis.model, Dataset.load_sample_dataset(), epoch)
        print('saved')
        print('time: ', time.time() - start, '\n')

        if hp.eval_model and (epoch + 1) % hp.epoch_per_eval == 0:
            print('evaluating...')
            start = time.time()
            eval_results = Evaluate.eval(dis.model, gen.model, test_dataset)
            for key in train_results:
                try:
                    results[key].append(train_results[key])
                except KeyError:
                    results[key] = [train_results[key]]
            for key in eval_results:
                try:
                    results[key].append(eval_results[key])
                except KeyError:
                    results[key] = [eval_results[key]]

            print('evaluated')
            print('time: ', time.time() - start, '\n')
            if not os.path.exists('results/figures'):
                os.makedirs('results/figures')
            for key in results:
                np.savetxt('results/figures/%s.txt' % key, results[key], fmt='%f')
                plt.title(key)
                plt.xlabel('Epochs')
                plt.ylabel(key)
                plt.plot([(i + 1) * hp.epoch_per_eval for i in range(len(results[key]))], results[key])
                plt.savefig('results/figures/%s.png' % key)
                plt.clf()

        dis.to_train()
        if not hp.train_only_enc:
            gen.to_train()


train_gan()

