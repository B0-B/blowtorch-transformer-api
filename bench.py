#!/usr/bin/env python3

'''
LLaMA2 benchmark example.
'''

from blowtorch import client

cl = client(hugging_face_path='TheBloke/Llama-2-7b-Chat-GPTQ', 
                name='GPT',
                device='gpu', 
                model_type="llama",
                trust_remote_code=False,
                revision="main")

cl.bench(tokens=512)

