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