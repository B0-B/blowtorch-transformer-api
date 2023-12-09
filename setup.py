#!/usr/bin/env python3

from os import system as shell

shell('pip install transformers[torch] ctransformers accelerate optimum h5py')
pip install --upgrade-strategy eager optimum[amd]
# shell('pip install --upgrade-strategy eager optimum[neural-compressor]')

# # tensorflow-intel dependecies
# dependencies = [
#     ('flatbuffers', '==23.1.21'),
#     ('google-pasta', '==0.1.1'),
#     ('grpcio', '==1.24.3'),
#     ('h5py', '==3.0.0'),
#     ('libclang', '==13.0.0'),
#     ('opt-einsum>=2.3.2', ''),
#     ('tensorboard', '==2.13'),
#     ('tensorflow-estimator', '==2.13.0'),
#     ('termcolor', '==1.1.0'),
#     ('gast', '==0.4.0'),
#     ('keras', '==2.13.1'),
#     ('numpy', '==1.24.3'),
#     ('typing-extensions', '==4.0.0')
# ]

# # substitute existing packages with the necessary versions
# for module, version in dependencies:
#     try:
#         shell(f'pip uninstall {module} -y')
#         print(f'Successfully removed {module}.')
#     except:
#         pass
#     try:
#         shell(f'pip install {module}{version}')
#         print(f'Successully installed {module}{version}')
#     except:
#         pass