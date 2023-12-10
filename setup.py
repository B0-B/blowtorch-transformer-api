#!/usr/bin/env python3

'''
flashtorch installation.


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
                      _.----.
    .----------------" /  /  \\
   (     flashtorch | |   |) |
    `----------------._\  \  /
                       "----'
    
   Fast Transformer & GPT API.
      
''')
print(f' ---- flashtorch {version} setup ---- ')

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
    name='flashtorch',
    version='1.1.0',
    author='B0-B',
    url='https://github.com/B0-B/flashtorch',
    packages=['flashtorch'],
    # py_modules=['flashtorch.py'],
    install_requires=[
        'transformers[torch]',
        'ctransformers',
        'accelerate',
        optimum_pkg,
        'h5py'
    ]
)
print('info: done.')