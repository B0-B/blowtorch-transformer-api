#!/usr/bin/env python3

'''
Llama-3-8B GPU benchmark example.
'''

from blowtorch import client

cl = client(hugging_face_path='MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ', 
            name='Assistant',
            device='gpu',
            context_length = 6000,
            chat_format="llama-3")

cl.bench(tokens=512)