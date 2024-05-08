#!/usr/bin/env python3

'''
Llama-3-70B GPU benchmark example.
'''

from blowtorch import client

cl = client(hugging_face_path='NousResearch/Meta-Llama-3-70B', 
            name='LlamaGPT',
            device='gpu',
            chat_format="llama-3")

cl.bench(tokens=512)