import os
from pathlib import Path
import logging
_logger = logging.getLogger(__file__)  # noqa

import numpy as np
try:
    import imageio
except ImportError:
    _logger.warning('Error importing package imageio')


data_directory = Path(os.environ['UMAP_CDW_DATA'])


def _to_numpy(path: os.PathLike, dtype=np.float32):
    try:
        with open(path, 'rb') as fh:
            r = np.fromfile(fh, dtype=dtype)
        return r.reshape(-1, 256, 256)
    except Exception as e:
        _logger.warning(f'File could not be loaded: {path.name}')
        _logger.warning(f'Exception was: {e}')
        return


def load(pattern: str, shape: tuple, dtype=np.float32):
    data = []
    for path in sorted(data_directory.glob(pattern)):
        s = _to_numpy(path, dtype)
        if s.shape != shape:
            _logger.warning(f'File could not be loaded: {path.name}')
            _logger.warning(f'Expected shape {shape}, got {s.shape}')
            continue
        if s is not None:
            data.append(s)
            continue
    return np.array(data)


def to_video(arr, fname, fps):
    writer = imageio.get_writer(fname, fps=fps)
    arr -= np.amin(arr)
    arr /= np.amax(arr)
    arr *= 255
    arr = arr.astype(np.uint8)
    for i in range(arr.shape[0]):
        writer.append_data(arr[i, :, :])
    writer.close()
