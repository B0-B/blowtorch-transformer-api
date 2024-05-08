#!/usr/bin/env python3
#
# blowtorch auto-install script. Will install provided wheel in ./dist branch.
# https://github.com/B0-B/blowtorch-transformer-api



from os import system as shell, chdir
from pathlib import Path

__root__ = Path(__file__).parent
chdir(__root__)

# read package version
with open(__root__.joinpath('version')) as f:
    version = f.read()
print(f'==== 🔥 blowtorch {version} 🔥 ====')

# try to import torch
try:
    import torch
except ImportError:
    print('[error] blowtorch demands an installed torch build, to ensure that the specific backend is setup like drivers etc.\nPlease install torch (pip install torch) and run again.')
    exit()

# find wheel file and install with pip manager
wheel = './dist/' + list(__root__.glob('./dist/*.whl'))[0].name
shell(f'pip install {wheel}')

# check for GPU specific backend
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    if 'amd' in device_name.lower():
        print(f'[GPU] AMD device detected: {device_name}')
    else:
        # assume NVIDIA otherwise
        print(f'[GPU] NVIDIA device detected: {device_name}')