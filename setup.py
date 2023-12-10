#!/usr/bin/env python3

'''
blowtorch installation.
'''

import sys
from os import system as shell
from distutils.core import setup

# derive package version
# with open(__root__.joinpath('version')) as f:
#     version = f.read()

# read arguments
arg = sys.argv[1]
version = sys.argv[2]

print('''\

            blowtorch
    
   Fast Transformer & GPT API.
      
''')
print(f' ---- blowtorch {version} setup ---- ')

# choose cuda or rocm packages
if arg == 'cuda':
    optimum_pkg = 'optimum'
elif arg == 'rocm':
    optimum_pkg = 'optimum[amd]'
else:
    optimum_pkg = ''
print(f'info: select {arg} packages.')

# start the setup
shell('python -m pip install --upgrade pip wheel')
setup(
    name='blowtorch',
    version='1.1.0',
    author='B0-B',
    url='https://github.com/B0-B/blowtorch',
    packages=['blowtorch'],
    # py_modules=['blowtorch.py'],
    install_requires=[
        'transformers[torch]',
        'ctransformers',
        'accelerate',
        optimum_pkg,
        'h5py'
    ]
)
print('info: done.')