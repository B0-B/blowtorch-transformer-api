#!/usr/bin/env python3
#
# blowtorch auto-install script. Will install provided wheel in ./dist branch.
# https://github.com/B0-B/blowtorch-transformer-api


import platform
from os import system as shell, chdir
from pathlib import Path

__root__ = Path(__file__).parent
chdir(__root__)

# read package version
with open(__root__.joinpath('version')) as f:
    version = f.read()
print(f'==== 🔥 blowtorch {version} 🔥 ====')
# find wheel file and instal with pip manager
wheel = './dist/' + list(__root__.glob('./dist/*.whl'))[0].name
shell(f'pip install {wheel}')