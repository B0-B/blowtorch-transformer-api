#!/usr/bin/env python3

'''
A demo of the Llama 3.2 release with 3B model.
Will be reachable at http://localhost:3000/
'''

from blowtorch import client, webUI, console

USERNAME = 'User'
ASSISTANT_NAME = 'Assistant'

cl = client('Llama-3.2-3B-Instruct.Q3_K_L.gguf', 
            'MaziyarPanahi/Llama-3.2-3B-Instruct-GGUF', 
            name=ASSISTANT_NAME,
            device='cpu',
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
    max_new_tokens=512,
    auto_trim = True
)

# expose chat to browser
console(cl)