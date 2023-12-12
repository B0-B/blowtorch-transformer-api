#!/usr/bin/env python3

'''
Chat with a sophisticated AI assistant.
'''

from blowtorch import client

bt = client('llama-2-7b-chat.Q2_K.gguf', 
    'TheBloke/Llama-2-7B-Chat-GGUF', 
    name='AI Assistant',
    device='cpu', 
    model_type="llama",
    max_new_tokens = 1000,
    context_length = 6000
)
bt.chat(
    max_new_tokens=128, 
    charTags=[
        'polite',
        'focused and helpful',
        'expert in programing',
        'obedient'
    ], 
    username='Human',
    do_sample=False, 
    temperature=0.8, 
    repetition_penalty=1.1
)
# bt.tokenizer.tokenize("This is a test sentence")