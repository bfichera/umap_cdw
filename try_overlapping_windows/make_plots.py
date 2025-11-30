from pathlib import Path
import pickle
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true')
do_show = parser.parse_args().show
if do_show:
    show = plt.show
else:

    def show():
        pass


for path in (Path.cwd() / 'results').glob('test2_results_024.pkl'):
    plots_folder = Path.cwd() / 'plots' / path.stem
    plots_folder.mkdir(parents=True, exist_ok=True)
    with open(path, 'rb') as fh:
        r = pickle.load(fh)
    plt.imshow(r.umap_low_res_rgb[0, :, :, :])
    plt.savefig(plots_folder / 'low_res_rgb')
    show()
    plt.imshow(r.umap_rgb[0, :, :, :])
    plt.savefig(plots_folder / 'rgb')
    show()
    plt.imshow(r.img_stk[0, 0, :, :])
    plt.savefig(plots_folder / 'img_stk0')
    show()
    plt.imshow(r.img_stk[0, 1, :, :])
    plt.savefig(plots_folder / 'img_stk1')
    show()
    plt.imshow(
        np.absolute(np.fft.fftshift(np.fft.fft2(r.img_stk[0, 0, :, :])))
    )
    plt.savefig(plots_folder / 'fft_img_stk0')
    show()
    plt.imshow(
        np.absolute(np.fft.fftshift(np.fft.fft2(r.img_stk[0, 1, :, :])))
    )
    plt.savefig(plots_folder / 'fft_img_stk1')
    show()

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    collapsed_rgb = r.umap_low_res_rgb.reshape(400, 3)
    ax.scatter(*tuple(collapsed_rgb[:, i] for i in range(3)))
    ax.set_xlabel('r')
    ax.set_ylabel('g')
    ax.set_zlabel('b')
    plt.savefig(plots_folder / 'rgb_scatter')
    show()

    with open('rois.json', 'r') as fh:
        roi_info = json.load(fh)

    rois = []
    for cluster in roi_info:
        new_cluster = []
        for roi in cluster:
            new_cluster.append(
                [slice(roi[0][0], roi[0][1]),
                 slice(roi[1][0], roi[1][1])]
            )
        rois.append(new_cluster)
    num_clusters = len(rois)
    fig, axs = plt.subplots(nrows=2, ncols=num_clusters, squeeze=False)
    twotimes = []
    for c, cluster in enumerate(rois):
        img = np.ma.masked_array(r.umap_low_res_rgb[0, :, :, :], mask=True)
        for roi in cluster:
            img.mask[roi[0], roi[1], :] = False
        axs[0, c].imshow(img.filled(np.nan))
        s0, s1, s2, s3, s4 = r.window_ttcf.shape
        mask = img.mask[:, :, 0]
        twotimes.append(
            r.window_ttcf.reshape(s0 * s1 * s2, s3,
                                  s4)[~mask.flatten(), :, :].mean(axis=0)
        )
    for t, twotime in enumerate(twotimes):
        axs[1, t].imshow(twotime, vmin=0, vmax=2)
    plt.savefig(plots_folder / 'cluster_twotimes')
    show()
    plt.close()

    s0, s1, s2, s3, s4 = r.window_ttcf.shape
    plt.imshow(
        np.std(r.window_ttcf.reshape(s0, s1, s2, s3 * s4), axis=-1)[0, :, :]
    )
    plt.savefig(plots_folder / 'rms_contrast')
    show()

    t = r.window_ttcf.reshape(s0, s1, s2, s3 * s4)
    plt.imshow(
        (
            (t.max(axis=-1) - t.min(axis=-1)) /
            (t.max(axis=-1) + t.min(axis=-1))
        )[0, :, :]
    )
    plt.savefig(plots_folder / 'michelson_contrast')
    show()
