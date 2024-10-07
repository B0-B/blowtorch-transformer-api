#!/usr/bin/env python3

'''
Creates GPT chat in web browser.
Will be reachable at http://localhost:3000/
'''


from blowtorch import client, webUI

cl = client(model_file='Meta-Llama-3-8B-Instruct.Q2_K.gguf', 
            hugging_face_path='MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF', 
            chat_format="llama-3",
            name='GPT',
            device='cpu',
            context_length = 6000)

cl.setConfig(
    char_tags=[
        'skilled like chat GPT', 
        'high performance',
        'accurate', 
        'problem-solving',
        'answer in markdown format',
        'kind', 
        'eloquent',
        'genuine'
    ], 
    username='User',
    temperature=0.85, 
    repetition_penalty=1.15
)

# cl.chat()
webUI(cl, port=3000)