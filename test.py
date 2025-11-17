import logging
import sys
import time
import argparse

import torch
from numba import config
import numpy as np
from workflowrecorder import Recorder
from UMAP_RGB.utils.window import WindowMesh
from UMAP_RGB.networks.EfficientNet_model import EfficientEncoder
from UMAP_RGB.utils.UMAP_RGB import UMAP, UMAP_RGB_video

from umap_cdw import load, to_video


parser = argparse.ArgumentParser()
parser.add_argument('--quick-test')
quick_test = parser.parse_args().quick_test

start_time = time.time()

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stderr))
logger.setLevel(logging.INFO)

logger.info(f"Numba thread layer: {config.THREADING_LAYER}")
logger.info(f"Numba threads: {config.NUMBA_NUM_THREADS}")
logger.info(f"PyTorch intra-op threads: {torch.get_num_threads()}")
logger.info(f"PyTorch inter-op threads: {torch.get_num_interop_threads()}")


with Recorder('test2_results', 'test2_results.pkl') as recorder:
    img_stk = load('test2_*.bin', (2, 256, 256))
    to_video(img_stk[:, -2, :, :] % (2*np.pi), 'out.mp4', 30)
    recorder.register(img_stk)
    if quick_test:
        logger.warning('Using wrong size data!')
        umap_in = img_stk[:20, -1, ::4, ::4]
    else:
        umap_in = img_stk[:, -1, :, :]
    recorder.register(umap_in)
    window_shape = (umap_in.shape[0], 12, 12)
    recorder.register(window_shape)
    step_shape = (umap_in.shape[0], 6, 6)
    recorder.register(step_shape)
    windows = WindowMesh(umap_in, window_shape, step_shape)
    recorder.register(windows)

    model = EfficientEncoder(windows, umap_in)
    low_res_feature_map, upscaler = model.extract_embedding(full_output=False)

    umap = UMAP(low_res_feature_map, upscaler)
    umap.generate_rgb(sparsity_mult=20)

    recorder.register(
        umap.get_rgb()[0],
        name='umap_get_rgb_0',
        description='umap.get_rgb()[0]',
    )
    recorder.register(
        umap.get_rgb(),
        name='umap_get_rgb',
        description='umap.get_rgb()',
    )

    UMAP_RGB_video(umap, umap_in, fps=4, alpha=0.5)

logger.info(f'Python script finished after {time.time() - start_time} seconds.')
