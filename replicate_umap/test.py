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
parser.add_argument('--min_half_window_length', type=int, default=12)
parser.add_argument('--max_half_window_length', type=int, default=12)
parser.add_argument('--window_length_steps', type=int, default=1)

_cfg = parser.parse_args()
quick_test = _cfg.quick_test
window_lengths = np.linspace(
    _cfg.min_half_window_length,
    _cfg.max_half_window_length,
    _cfg.window_length_steps,
    endpoint=True
).astype(int) * 2

start_time = time.time()

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stderr))
logger.setLevel(logging.INFO)

logger.info(f"Numba thread layer: {config.THREADING_LAYER}")
logger.info(f"Numba threads: {config.NUMBA_NUM_THREADS}")
logger.info(f"PyTorch intra-op threads: {torch.get_num_threads()}")
logger.info(f"PyTorch inter-op threads: {torch.get_num_interop_threads()}")

for window_length in window_lengths:
    iteration_start_time = time.time()
    with Recorder(
            f'test2_results_{window_length:03d}',
            f'test2_results_{window_length:03d}.pkl',
    ) as recorder:
        img_stk = load('test2_*.bin', (2, 256, 256))
        to_video(img_stk[:, -2, :, :] % (2 * np.pi), 'out.mp4', 30)
        recorder.register(img_stk)
        if quick_test:
            logger.warning('Using wrong size data!')
            umap_in = img_stk[:20, -1, ::4, ::4]
        else:
            umap_in = img_stk[:, -1, :, :]
        recorder.register(umap_in)
        window_shape = (umap_in.shape[0], window_length, window_length)
        recorder.register(window_shape)
        step_shape = (umap_in.shape[0], window_length // 2, window_length // 2)
        recorder.register(step_shape)
        windows = WindowMesh(umap_in, window_shape, step_shape)
        recorder.register(windows)

        model = EfficientEncoder(windows, umap_in)
        low_res_feature_map, upscaler = model.extract_embedding(
            full_output=False
        )

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

    logger.info(
        'Single window length finished after '
        f'{time.time() - iteration_start_time} seconds.'
    )

logger.info(
    f'Entire script finished after {time.time() - start_time} seconds.'
)
