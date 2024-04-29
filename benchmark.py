#!/usr/bin/env python3

'''
LLaMA2 benchmark example.
'''

from blowtorch import client

cl = client(hugging_face_path='NousResearch/Llama-2-7b-chat-hf', 
                name='GPT',
                device='gpu', 
                model_type="llama")

cl.bench(tokens=512)