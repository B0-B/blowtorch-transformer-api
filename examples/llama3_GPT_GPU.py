#!/usr/bin/env python3

'''
An example of LLaMA-3-8B GPTQ model (suitable for GPU) with GPT-like chat.
The GPT will refer to the user name provided.
Will be reachable at http://localhost:3000/
'''

from blowtorch import client, webUI, console

USERNAME = input('Please enter your name:')

cl = client(hugging_face_path='MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ', 
            name='Assistant',
            device='gpu',
            chat_format="llama-3")

cl.setConfig(
    char_tags=[
        'skilled like chat GPT', 
        'high performance',
        'accurate', 
        'problem-solving',
        'eloquent',
        'genuine'
    ], 
    username=USERNAME,
    temperature=1.1, 
    max_new_tokens=256
)

# expose chat to browser
console(cl)