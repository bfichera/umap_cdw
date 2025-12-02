import logging
import sys
import time
import argparse
from pathlib import Path
import itertools

import torch
from numba import config
from workflowrecorder import Recorder
from UMAP_RGB.utils.window import WindowMesh
from UMAP_RGB.networks.EfficientNet_model import EfficientEncoder
from UMAP_RGB.utils.UMAP_RGB import UMAP

from umap_cdw import load

parser = argparse.ArgumentParser()
parser.add_argument('--quick-test', action='store_true')
parser.add_argument('--half-window-lengths', type=int, nargs='+')
parser.add_argument('--window-stepsize-ratios', type=int, nargs='+')

_cfg = parser.parse_args()
quick_test = _cfg.quick_test
window_lengths = [c * 2 for c in _cfg.half_window_lengths]
window_stepsize_ratios = _cfg.window_stepsize_ratios

start_time = time.time()

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stderr))
logger.setLevel(logging.INFO)

logger.info(f"Numba thread layer: {config.THREADING_LAYER}")
logger.info(f"Numba threads: {config.NUMBA_NUM_THREADS}")
logger.info(f"PyTorch intra-op threads: {torch.get_num_threads()}")
logger.info(f"PyTorch inter-op threads: {torch.get_num_interop_threads()}")

results_path = Path.cwd() / 'results'
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)
for window_length, window_stepsize_ratio in itertools.product(
        window_lengths, window_stepsize_ratios):
    iteration_start_time = time.time()
    with Recorder(
            f'test2_results_{window_length:03d}_{window_stepsize_ratio:03d}',
            results_path / f'test2_results_{window_length:03d}_{window_stepsize_ratio:03d}.pkl',
    ) as recorder:
        img_stk = load('test2_*.bin', (2, 256, 256))
        recorder.register(img_stk)
        if quick_test:
            logger.warning('Using wrong size data!')
            umap_in = img_stk[:20, -1, ::4, ::4]
        else:
            umap_in = img_stk[:, -1, :, :]
        recorder.register(umap_in)
        window_shape = (umap_in.shape[0], window_length, window_length)
        recorder.register(window_shape)
        step_shape = (
            umap_in.shape[0], window_length // window_stepsize_ratio,
            window_length // window_stepsize_ratio
        )
        recorder.register(step_shape)
        windows = WindowMesh(umap_in, window_shape, step_shape)

        model = EfficientEncoder(windows, umap_in)
        low_res_feature_map, upscaler = model.extract_embedding(
            full_output=False
        )

        recorder.register(windows.window_ttcf, name='window_ttcf')

        umap = UMAP(low_res_feature_map, upscaler)
        umap.generate_rgb(sparsity_mult=20)

        recorder.register(
            umap.rgb[0],
            name='umap_get_rgb_0',
            description='umap.rgb[0]',
        )
        recorder.register(
            umap.rgb,
            name='umap_rgb',
            description='umap.rgb',
        )
        recorder.register(
            umap.low_res_rgb,
            name='umap_low_res_rgb',
            description='umap.low_res_rgb',
        )

    logger.info(
        'Single window length / stepsize combination finished after '
        f'{time.time() - iteration_start_time} seconds.'
    )

logger.info(
    f'Entire script finished after {time.time() - start_time} seconds.'
)
