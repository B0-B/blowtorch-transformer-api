#!/usr/bin/env python3

'''
blowtorch installation.
'''

import sys
from os import system as shell
from setuptools import setup
from pathlib import Path

# derive package version
__root__ = Path(__file__).parent
with open(__root__.joinpath('version')) as f:
    version = f.read()

# read arguments
print('arguments', sys.argv)
arg = sys.argv[2]
# version = sys.argv[3]
sys.argv.remove(arg)

print('''\

            blowtorch
    
   Fast Transformer & GPT API.
      
''')
print(f' ---- blowtorch {version} setup ---- ')

# choose cuda or rocm packages

print(f'info: select {arg} packages.')

# start the setup
shell('python -m pip install --upgrade pip wheel')
setup(
    name='blowtorch',
    version=version,
    author='B0-B',
    url='https://github.com/B0-B/blowtorch',
    packages=['blowtorch'],
    # py_modules=['blowtorch.py'],
    install_requires=[
        'transformers==4.31.0',
        'ctransformers==0.2.27',
        'accelerate==0.21.0',
        'h5py==3.9.0'
    ]
)

# gfx packages
if arg == 'cuda':
    shell('pip install optimum')
elif arg == 'rocm':
    shell('pip install --upgrade-strategy eager optimum[amd]')
else:
    pass

print('info: done.')