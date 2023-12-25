#!/usr/bin/env python3

'''
Creates GPT chat in web browser.
Will be reachable at http://localhost:3000/
'''


from blowtorch import client, webUI

cl = client('llama-2-7b-chat.Q2_K.gguf', 
            'TheBloke/Llama-2-7B-Chat-GGUF', 
            name='GPT',
            device='cpu', 
            model_type="llama",
            max_new_tokens = 1000,
            context_length = 6000)

cl.setConfig(
    char_tags=[
        'skilled like chat GPT', 
        'high performance',
        'accurate', 
        'problem-solving',
        'kind', 
        'eloquent',
        'genuine'
    ], 
    username='User',
    do_sample=True, 
    temperature=0.85, 
    repetition_penalty=1.15
)

# cl.chat()
webUI(cl, port=3000)