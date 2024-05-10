#!/usr/bin/env python3

'''
Llama-3-8B CPU benchmark example.
'''

from blowtorch import client

cl = client('Meta-Llama-3-8B-Instruct.Q3_K_L.gguf', 
            'QuantFactory/Meta-Llama-3-8B-Instruct-GGUF', 
            name='LlamaGPT',
            device='cpu',
            chat_format="llama-3")

cl.bench(tokens=512)