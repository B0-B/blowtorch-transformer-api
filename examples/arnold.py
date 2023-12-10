#!/usr/bin/env python3

'''
Chat with the actor, body-building legend, and former governer himself!
'''

from flashtorch import flashModel

flashModel('llama-2-7b-chat.Q2_K.gguf', 
           'TheBloke/Llama-2-7B-Chat-GGUF', 
           name='Arnold',
           device='cpu', 
           model_type="llama",
           max_new_tokens = 1000,
           context_length = 6000
).chat(
    max_new_tokens=128, 
    charTags=[
        'imressionating Arnold Schwarzenegger',
        'with thick austrian accent', 
        'answers with famous phrases from his movies',
        'a bit arrogantly',
        'funnily annoyed',
        'jokes around',
        'complaining about his current age'
    ], 
    do_sample=False, 
    temperature=0.8, 
    repetition_penalty=1.1
)