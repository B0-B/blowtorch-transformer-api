#!/usr/bin/env python3

'''
An example of LLaMA-3-8B GGUF model with GPT-like chat.
The GPT can chat with a user whose name can be provided.
Will be reachable at http://localhost:3000/
'''

from blowtorch import client, webUI, console

USERNAME = 'Steve'

cl = client('mistral-7b-instruct-v0.2.Q6_K.gguf', 
            'TheBloke/Mistral-7B-Instruct-v0.2-GGUF', 
            name='Llama-GPT',
            device='cpu',
            chat_format="llama-2")

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
webUI(cl)