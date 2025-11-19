import pickle

import numpy as np
import matplotlib.pyplot as plt


with open('results.pkl', 'rb') as fh:
    r = pickle.load(fh)

plt.imshow(np.sum(r.img_stk[:, 0, :, :], axis=0))
plt.savefig('img_stk0')
plt.imshow(np.sum(r.img_stk[:, 1, :, :], axis=0))
plt.savefig('img_stk1')
plt.imshow(r.umap_get_rgb_0)
plt.savefig('umap_get_rgb_0')
