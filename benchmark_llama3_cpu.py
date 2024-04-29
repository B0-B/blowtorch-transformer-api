#!/usr/bin/env python3

'''
LLaMA2 benchmark example.
'''

from blowtorch import client

cl = client('Meta-Llama-3-8B-Instruct.Q3_K_L.gguf', 
            'QuantFactory/Meta-Llama-3-8B-Instruct-GGUF', 
            name='Assistant',
            device='cpu',
            context_length = 6000,
            chat_format="llama-3")

cl.bench(tokens=64)