import logging
import sys

import numpy as np
from workflowrecorder import Recorder
from UMAP_RGB.utils.window import WindowMesh
from UMAP_RGB.networks.EfficientNet_model import EfficientEncoder
from UMAP_RGB.utils.UMAP_RGB import UMAP, UMAP_RGB_video


from umap_cdw import load, to_video


logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stderr))


with Recorder('test2_results', 'test2_results.pkl') as recorder:
    img_stk = load('test2_*.bin', (2, 256, 256))
    to_video(img_stk[:, -2, :, :] % (2*np.pi), 'out.mp4', 30)
    recorder.register(img_stk)
    umap_in = img_stk[:, -1, :, :]
    recorder.register(umap_in)
    window_shape = (umap_in.shape[0], 12, 12)
    recorder.register(window_shape)
    step_shape = (None, 6, 6)
    recorder.register(step_shape)
    windows = WindowMesh(umap_in, window_shape, step_shape)
    recorder.register(windows)

    model = EfficientEncoder(windows, img_stk)
    feature_map, low_res_feature_map, upscaler = model.extract_embedding()

    umap = UMAP(low_res_feature_map, upscaler)
    umap.generate_rgb(sparsity_mult=20)
    recorder.register(umap)

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

    UMAP_RGB_video(umap, img_stk, fps=4, alpha=0.5)
