#!/usr/bin/env python3
#
# blowtorch auto-build script. Will build a wheel file in ./dist directory.
# https://github.com/B0-B/blowtorch-transformer-api

import platform, torch
from os import system as shell, chdir
from pathlib import Path

__root__ = Path(__file__).parent
chdir(__root__)

# gather all device names
devices = [torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())]

if platform.system() == 'Windows':
    # Make sure the latest version of PyPA’s build is installed
    shell('python -m pip install --upgrade pip build')
    shell('python -m build')
elif platform.system() in ['Linux', 'Darwin']:
    # Make sure the latest version of PyPA’s build is installed
    shell('python3 -m pip install --upgrade pip build')
    shell('python3 -m build')
else:
    print('Unknown OS')

# Re-install llama-cpp with acceleration backend
# if not devices:
#     ENV = 'CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"'
# elif 'amd' in devices[0].lower():
#     ENV = 'CMAKE_ARGS="-DLLAMA_HIPBLAS=on"'
# elif 'nvidia' in devices[0].lower():
#     ENV = 'CMAKE_ARGS="-DLLAMA_CUDA=on"'

# ENV = f'$env:{ENV}' if platform.system() == 'Windows' else ENV

# shell(f'{ENV} pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir')