"""Observed execution environment, captured before fitting without changing it."""
from functools import cache
import os
import platform
import subprocess
import sys


@cache
def capture_environment() -> dict:
    import torch

    freeze = subprocess.run([sys.executable, '-m', 'pip', '--disable-pip-version-check',
                             'freeze', '--all'], check=True, capture_output=True, text=True)
    return {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
            'pip_freeze': freeze.stdout.splitlines(), 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'thread_env': {key: value for key, value in sorted(os.environ.items())
                           if 'THREAD' in key or key.startswith(('OMP_', 'MKL_', 'OPENBLAS_', 'NUMEXPR_',
                               'CUDA_', 'CUBLAS_', 'CUDNN_', 'PYTORCH_', 'NVIDIA_', 'VECLIB_', 'BLIS_'))}}
