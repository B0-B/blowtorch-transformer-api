#!/usr/bin/env python3

'''
LLaMA2 benchmark example.
'''

from blowtorch import client

client(hugging_face_path='TheBloke/Llama-2-7B-Chat-GGML', device='cpu', model_type="llama").bench() 