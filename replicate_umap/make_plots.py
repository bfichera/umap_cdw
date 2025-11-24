import pickle
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt

pathd = {}
for path in sorted(Path.cwd().glob('*results*.pkl')):
    try:
        fname = path.name
        prog = re.compile(r'^test2_results_([0-9]*).pkl$')
        window_length = int(prog.match(fname).groups()[0])
        pathd[window_length] = path
    except AttributeError:
        continue

for window_length, path in pathd.items():
    with open(path, 'rb') as fh:
        r = pickle.load(fh)
    folder = Path.cwd() / f'plots/plots_{window_length:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    plt.imshow(np.sum(r.img_stk[:, 0, :, :], axis=0))
    plt.savefig(folder / 'img_stk0')
    plt.imshow(np.sum(r.img_stk[:, 1, :, :], axis=0))
    plt.savefig(folder / 'img_stk1')
    plt.imshow(r.umap_get_rgb_0)
    plt.savefig(folder / 'umap_get_rgb_0')
