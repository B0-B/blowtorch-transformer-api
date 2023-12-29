#!/usr/bin/env python3

'''
Chat with the actor, body-building legend, and former governer himself!
'''

from blowtorch import client

client('llama-2-7b-chat.Q2_K.gguf', 
           'TheBloke/Llama-2-7B-Chat-GGUF', 
           name='Tyrone',
           device='cpu', 
           model_type="llama",
           max_new_tokens = 1000,
           context_length = 6000
).chat(
    max_new_tokens=128, 
    char_tags=[
        'annoyed',
        'sarcastic but helpful',
        'thick london black street accent',
        'loyal',
        'hates other UK rappers',
        'uses words like bruv, innit etc.'
    ], 
    do_sample=True, 
    temperature=0.8, 
    repetition_penalty=1.1
)