from pathlib import Path
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=lambda s: Path(s))
parser.add_argument('--show', action='store_true')
parser.add_argument('--output-dir', type=lambda s: Path(s))
parser.add_argument('--extension', type=str, default='.pdf')
_cfg = parser.parse_args()
do_show = _cfg.show
output_dir = _cfg.output_dir
results_path = _cfg.results_path
extension = _cfg.extension

if do_show:
    show = plt.show
else:

    def show():
        pass

plots_folder = _cfg.output_dir
with open(results_path, 'rb') as fh:
    r = pickle.load(fh)
plt.imshow(r.mapper_low_res_rgb[0, :, :, :])
plt.savefig(plots_folder / f'low_res_rgb{extension}')
show()
plt.imshow(r.mapper_rgb[0, :, :, :])
plt.savefig(plots_folder / f'rgb{extension}')
show()
plt.imshow(r.img_stk[0, 0, :, :])
plt.savefig(plots_folder / f'img_stk0{extension}')
show()
plt.imshow(r.img_stk[0, 1, :, :])
plt.savefig(plots_folder / f'img_stk1{extension}')
show()
plt.imshow(np.sum(r.img_stk[:, 0, :, :], axis=0))
plt.savefig(plots_folder / f'sum_img_stk0{extension}')
show()
plt.imshow(np.sum(r.img_stk[:, 1, :, :], axis=0))
plt.savefig(plots_folder / f'sum_img_stk1{extension}')
show()
plt.imshow(
    np.absolute(np.fft.fftshift(np.fft.fft2(r.img_stk[0, 0, :, :])))
)
plt.savefig(plots_folder / f'fft_img_stk0{extension}')
show()
plt.imshow(
    np.absolute(np.fft.fftshift(np.fft.fft2(r.img_stk[0, 1, :, :])))
)
plt.savefig(plots_folder / f'fft_img_stk1{extension}')
show()

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
u0, u1, u2, u3 = r.mapper_low_res_rgb.shape
collapsed_rgb = r.mapper_low_res_rgb.reshape(u1 * u2, 3)
ax.scatter(*(collapsed_rgb[:, i] for i in range(3)))
ax.set_xlabel('r')
ax.set_ylabel('g')
ax.set_zlabel('b')
plt.savefig(plots_folder / f'rgb_scatter{extension}')
show()
plt.close()

s0, s1, s2, s3, s4 = r.window_ttcf.shape
plt.imshow(
    np.std(r.window_ttcf.reshape(s0, s1, s2, s3 * s4), axis=-1)[0, :, :]
)
plt.savefig(plots_folder / f'rms_contrast{extension}')
show()
plt.close()

t = r.window_ttcf.reshape(s0, s1, s2, s3 * s4)
plt.imshow(
    (
        (t.max(axis=-1) - t.min(axis=-1)) /
        (t.max(axis=-1) + t.min(axis=-1))
    )[0, :, :]
)
plt.savefig(plots_folder / f'michelson_contrast{extension}')
show()
plt.close()
