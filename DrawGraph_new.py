import matplotlib.pyplot as plt
import os


def file_to_array(path):
    with open(path) as f:
        return [float(v) for v in f.readlines()]


def _graph(baseline, mse_d, mse_dg, dls_d, dls_dg, title, y_label, file_name, ylim=None):
    epochs = [i+1 for i in range(len(dls_d))]
    plt.title(title)
    if baseline is not None:
        plt.plot(epochs, baseline, label='Baseline', color='tab:green')
    if mse_d is not None:
        plt.plot(epochs, mse_d, label='No DLS, D', color='tab:red')
    if mse_dg is not None:
        plt.plot(epochs, mse_dg, label='No DLS, DG', color='tab:blue')
    plt.plot(epochs, dls_d, label='DLS, D', color='tab:orange')
    plt.plot(epochs, dls_dg, label='DLS, DG', color='tab:brown')
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.legend()
    if ylim is not None:
        plt.ylim(ylim)

    if not os.path.exists('./results'):
        os.makedirs('./results')
    plt.savefig('./results/' + file_name + '.png')
    plt.clf()


baseline_path = r'D:\DLSGAN_Results\Baseline_results_combine'
mse_d_path = r'D:\DLSGAN_Results\N_MSE_D'
mse_dg_path = r'D:\DLSGAN_Results\N_MSE_DG'
dls_d_path = r'D:\DLSGAN_Results\N_DLS_D'
dls_dg_path = r'D:\DLSGAN_Results\N_DLS_DG'


def draw_graphs():
    _graph(
        file_to_array(baseline_path + r'\results\fids.txt'),
        file_to_array(mse_d_path + r'\results\fids.txt'),
        file_to_array(mse_dg_path + r'\results\fids.txt'),
        file_to_array(dls_d_path + r'\results\fids.txt'),
        file_to_array(dls_dg_path + r'\results\fids.txt'),
        'Generative performance',
        'FID',
        'FID',
        (0, 100)
    )
    _graph(
        file_to_array(baseline_path + r'\results\Precisions.txt'),
        file_to_array(mse_d_path + r'\results\Precisions.txt'),
        file_to_array(mse_dg_path + r'\results\Precisions.txt'),
        file_to_array(dls_d_path + r'\results\Precisions.txt'),
        file_to_array(dls_dg_path + r'\results\Precisions.txt'),
        'Generative performance',
        'Precision',
        'Precision'
    )
    _graph(
        file_to_array(baseline_path + r'\results\Recalls.txt'),
        file_to_array(mse_d_path + r'\results\Recalls.txt'),
        file_to_array(mse_dg_path + r'\results\Recalls.txt'),
        file_to_array(dls_d_path + r'\results\Recalls.txt'),
        file_to_array(dls_dg_path + r'\results\Recalls.txt'),
        'Generative performance',
        'Recall',
        'Recall'
    )
    _graph(
        file_to_array(baseline_path + r'\results\enc_losses.txt'),
        file_to_array(mse_d_path + r'\results\enc_losses.txt'),
        file_to_array(mse_dg_path + r'\results\enc_losses.txt'),
        file_to_array(dls_d_path + r'\results\enc_losses.txt'),
        file_to_array(dls_dg_path + r'\results\enc_losses.txt'),
        'Average encoder loss',
        '$L_{enc}$',
        'L_enc',
        (0, 1.5)
    )
    _graph(
        file_to_array(baseline_path + r'\results\fake_psnrs.txt'),
        file_to_array(mse_d_path + r'\results\fake_psnrs.txt'),
        file_to_array(mse_dg_path + r'\results\fake_psnrs.txt'),
        file_to_array(dls_d_path + r'\results\fake_psnrs.txt'),
        file_to_array(dls_dg_path + r'\results\fake_psnrs.txt'),
        'Inversion performance',
        'Fake PSNR',
        'fake_psnr'
    )
    _graph(
        file_to_array(baseline_path + r'\results\fake_ssims.txt'),
        file_to_array(mse_d_path + r'\results\fake_ssims.txt'),
        file_to_array(mse_dg_path + r'\results\fake_ssims.txt'),
        file_to_array(dls_d_path + r'\results\fake_ssims.txt'),
        file_to_array(dls_dg_path + r'\results\fake_ssims.txt'),
        'Inversion performance',
        'Fake SSIM',
        'fake_ssim'
    )
    _graph(
        file_to_array(baseline_path + r'\results\real_psnrs.txt'),
        file_to_array(mse_d_path + r'\results\real_psnrs.txt'),
        file_to_array(mse_dg_path + r'\results\real_psnrs.txt'),
        file_to_array(dls_d_path + r'\results\real_psnrs.txt'),
        file_to_array(dls_dg_path + r'\results\real_psnrs.txt'),
        'Comprehensive performance',
        'Real PSNR',
        'real_psnr'
    )
    _graph(
        file_to_array(baseline_path + r'\results\real_ssims.txt'),
        file_to_array(mse_d_path + r'\results\real_ssims.txt'),
        file_to_array(mse_dg_path + r'\results\real_ssims.txt'),
        file_to_array(dls_d_path + r'\results\real_ssims.txt'),
        file_to_array(dls_dg_path + r'\results\real_ssims.txt'),
        'Comprehensive performance',
        'Real SSIM',
        'real_ssim'
    )

    _graph(
        None,
        None,
        None,
        file_to_array(dls_d_path + r'\results\latent_entropys.txt'),
        file_to_array(dls_dg_path + r'\results\latent_entropys.txt'),
        'Differential entropy of scaled latent random variable',
        'Differential entropy',
        'entropy',
    )

draw_graphs()
