from pathlib import Path
import pickle

import matplotlib.pyplot as plt

for path in (Path.cwd() / 'results').glob('*.pkl'):
    plots_folder = Path.cwd() / 'plots' / path.stem
    plots_folder.mkdir(parents=True, exist_ok=True)
    with open(path, 'rb') as fh:
        r = pickle.load(fh)
    plt.imshow(r.umap_low_res_rgb[0, :, :, :])
    plt.savefig(plots_folder / 'low_res_rgb')
    plt.imshow(r.umap_rgb[0, :, :, :])
    plt.savefig(plots_folder / 'rgb')

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    collapsed_rgb = r.umap_low_res_rgb.reshape(400, 3)
    ax.scatter(*tuple(collapsed_rgb[:, i] for i in range(3)))
    ax.set_xlabel('r')
    ax.set_ylabel('g')
    ax.set_zlabel('b')
    plt.savefig(plots_folder / 'rgb_scatter')

    
