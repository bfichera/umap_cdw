import logging
import sys

import numpy as np
from workflowrecorder import Recorder
from UMAP_RGB.utils.window import WindowMesh
from UMAP_RGB.networks.EfficientNet_model import EfficientEncoder


from umap_cdw import load, to_video


logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stderr))


with Recorder('test2_results', 'test2_results.pkl') as recorder:
    img_stk = load('test2_*.bin', (2, 256, 256))
    to_video(img_stk[:, -2, :, :] % (2*np.pi), 'out.mp4', 30)
    recorder.register(img_stk)
    umap_in = img_stk[:, -1, :, :]
    window_shape = (umap_in.shape[0], 4, 4)
    step_shape = (None, 2, 2)
    windows = WindowMesh(umap_in, window_shape, step_shape)

    model = EfficientEncoder(windows, img_stk)
    feature_map, low_res_feature_map, upscaler = model.extract_embedding()
