#!/usr/bin/env python3

'''
Chat with the actor, body-building legend, and former governer himself!
'''

from blowtorch import client

client( model_file='Meta-Llama-3-8B-Instruct.Q2_K.gguf', 
        hugging_face_path='MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF', 
        chat_format="llama-3",
        name='Arnold',
        device='cpu', 
        max_new_tokens = 1000,
        context_length = 6000
).chat(
    max_new_tokens=128, 
    char_tags=[
        'imressionating Arnold Schwarzenegger',
        'with thick austrian accent', 
        'often answers with famous quotes from his movies',
        'references everything to himself',
        'a bit arrogantly',
        'funnily annoyed',
        'jokes around',
        'complaining about his current age'
    ], 
    temperature=0.8, 
    repetition_penalty=1.1
)