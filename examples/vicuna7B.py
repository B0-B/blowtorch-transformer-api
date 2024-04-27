#!/usr/bin/env python3

'''
How to get Vicuna-7B in GPTQ format to run.
Please see the client arguments.
'''

from blowtorch import client, webUI

cl = client(hugging_face_path='TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ', 
            name='GPT',
            device='cpu', 
            device_map="auto",
            trust_remote_code=True,
            revision="main")

cl.setConfig(
    username='User',
    temperature=0.8, 
    repetition_penalty=1.1
)

# cl.chat()
webUI(cl)