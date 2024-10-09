#!/usr/bin/env python3

'''
An example of LLaMA-3-8B GGUF model with GPT-like chat.
The GPT can chat with a user whose name can be provided.
Will be reachable at http://localhost:3000/
'''

from blowtorch import client, webUI, console

USERNAME = 'User'
ASSISTANT_NAME = 'Assistant'

cl = client('Meta-Llama-3-8B-Instruct.Q2_K.gguf', 
            'MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF', 
            name='ASSISTANT_NAME',
            device='cpu',
            chat_format="llama-3",
            n_ctx=2048)

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