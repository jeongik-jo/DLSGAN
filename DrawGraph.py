import matplotlib.pyplot as plt
import os
import numpy as np


def file_to_array(path):
    with open(path) as f:
        return [float(v) for v in f.readlines()]


def _graph(dls_d, dls_dg, mse_d, mse_dg, title, y_label, file_name, ylim=None):
    epochs = [i+1 for i in range(len(dls_d))]
    plt.title(title)
    plt.plot(epochs, dls_d, label='DLS, D')
    plt.plot(epochs, dls_dg, label='DLS, DG')
    if mse_d is not None:
        plt.plot(epochs, mse_d, label='No DLS, D')
    if mse_dg is not None:
        plt.plot(epochs, mse_dg, label='No DLS, DG')
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.legend()
    if ylim is not None:
        plt.ylim(ylim)

    if not os.path.exists('./results'):
        os.makedirs('./results')
    plt.savefig('./results/' + file_name + '.png')
    plt.clf()

#title = '$Z{\sim}N(0,1^2)$'
#title = '$Z{\sim}U(-\sqrt{3},\sqrt{3})$'
dls_d_paths = [r'D:\DLSGAN_Results\N_DLS_D_pr']
dls_dg_paths = [r'D:\DLSGAN_Results\N_DLS_DG_pr']
mse_d_paths = [r'D:\DLSGAN_Results\N_MSE_D_pr']
mse_dg_paths = [r'D:\DLSGAN_Results\N_MSE_DG_pr']


def draw_graphs():
    _graph(
        np.mean([file_to_array(path + r'\results\fids.txt') for path in dls_d_paths], axis=0),
        np.mean([file_to_array(path + r'\results\fids.txt') for path in dls_dg_paths], axis=0),
        np.mean([file_to_array(path + r'\results\fids.txt') for path in mse_d_paths], axis=0),
        np.mean([file_to_array(path + r'\results\fids.txt') for path in mse_dg_paths], axis=0),
        'Generative Performance of GAN',
        'FID',
        'FID',
        (0, 100)
    )
    _graph(
        np.mean([file_to_array(path + r'\results\Precisions.txt') for path in dls_d_paths], axis=0),
        np.mean([file_to_array(path + r'\results\Precisions.txt') for path in dls_dg_paths], axis=0),
        np.mean([file_to_array(path + r'\results\Precisions.txt') for path in mse_d_paths], axis=0),
        np.mean([file_to_array(path + r'\results\Precisions.txt') for path in mse_dg_paths], axis=0),
        'Generative Performance of GAN',
        'Precision',
        'Precision'
    )
    _graph(
        np.mean([file_to_array(path + r'\results\Recalls.txt') for path in dls_d_paths], axis=0),
        np.mean([file_to_array(path + r'\results\Recalls.txt') for path in dls_dg_paths], axis=0),
        np.mean([file_to_array(path + r'\results\Recalls.txt') for path in mse_d_paths], axis=0),
        np.mean([file_to_array(path + r'\results\Recalls.txt') for path in mse_dg_paths], axis=0),
        'Generative Performance of GAN',
        'Recall',
        'Recall'
    )
    _graph(
        np.mean([file_to_array(path + r'\results\enc_losses.txt') for path in dls_d_paths], axis=0),
        np.mean([file_to_array(path + r'\results\enc_losses.txt') for path in dls_dg_paths], axis=0),
        np.mean([file_to_array(path + r'\results\enc_losses.txt') for path in mse_d_paths], axis=0),
        np.mean([file_to_array(path + r'\results\enc_losses.txt') for path in mse_dg_paths], axis=0),
        'encoder loss',
        '$L_{enc}$',
        'L_enc',
        (0, 1.5)
    )
    _graph(
        np.mean([file_to_array(path + r'\results\fake_psnrs.txt') for path in dls_d_paths], axis=0),
        np.mean([file_to_array(path + r'\results\fake_psnrs.txt') for path in dls_dg_paths], axis=0),
        np.mean([file_to_array(path + r'\results\fake_psnrs.txt') for path in mse_d_paths], axis=0),
        np.mean([file_to_array(path + r'\results\fake_psnrs.txt') for path in mse_dg_paths], axis=0),
        'Reconstruction performance',
        'Fake PSNR',
        'fake_psnr'
    )
    _graph(
        np.mean([file_to_array(path + r'\results\fake_ssims.txt') for path in dls_d_paths], axis=0),
        np.mean([file_to_array(path + r'\results\fake_ssims.txt') for path in dls_dg_paths], axis=0),
        np.mean([file_to_array(path + r'\results\fake_ssims.txt') for path in mse_d_paths], axis=0),
        np.mean([file_to_array(path + r'\results\fake_ssims.txt') for path in mse_dg_paths], axis=0),
        'Reconstruction performance',
        'Fake SSIM',
        'fake_ssim'
    )
    _graph(
        np.mean([file_to_array(path + r'\results\real_psnrs.txt') for path in dls_d_paths], axis=0),
        np.mean([file_to_array(path + r'\results\real_psnrs.txt') for path in dls_dg_paths], axis=0),
        np.mean([file_to_array(path + r'\results\real_psnrs.txt') for path in mse_d_paths], axis=0),
        np.mean([file_to_array(path + r'\results\real_psnrs.txt') for path in mse_dg_paths], axis=0),
        'Reconstruction performance',
        'Real PSNR',
        'real_psnr'
    )
    _graph(
        np.mean([file_to_array(path + r'\results\real_ssims.txt') for path in dls_d_paths], axis=0),
        np.mean([file_to_array(path + r'\results\real_ssims.txt') for path in dls_dg_paths], axis=0),
        np.mean([file_to_array(path + r'\results\real_ssims.txt') for path in mse_d_paths], axis=0),
        np.mean([file_to_array(path + r'\results\real_ssims.txt') for path in mse_dg_paths], axis=0),
        'Reconstruction performance',
        'Real SSIM',
        'real_ssim'
    )

    _graph(
        np.mean([file_to_array(path + r'\results\latent_entropys.txt') for path in dls_d_paths], axis=0),
        np.mean([file_to_array(path + r'\results\latent_entropys.txt') for path in dls_dg_paths], axis=0),
        None,
        None,
        'Latent random variable entropy',
        'Latent entropy',
        'entropy',
        (0, 750)
    )

draw_graphs()
