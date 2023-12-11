#!/usr/bin/env python3

'''
Build install binaries and install package with pip.
'''

import sys
from os import system as shell
from pathlib import Path

__root__ = Path(__file__).parent
with open(__root__.joinpath('version')) as f:

    version = f.read()

shell(f'python setup.py sdist {sys.argv[1]}')
shell(f'pip install ./dist/blowtorch-{version}.tar.gz')