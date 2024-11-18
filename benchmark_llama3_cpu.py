#!/usr/bin/env python3

'''
Llama-3-8B CPU benchmark example.
'''

from blowtorch import client

cl = client(model_file='Meta-Llama-3-8B-Instruct.Q2_K.gguf', 
            hugging_face_path='MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF', 
            chat_format="llama-3",
            device="cpu")

cl.bench(tokens=512)