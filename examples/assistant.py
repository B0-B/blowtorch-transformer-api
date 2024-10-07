#!/usr/bin/env python3

'''
Chat with a sophisticated AI assistant.
'''

from blowtorch import client

client( model_file='Meta-Llama-3-8B-Instruct.Q2_K.gguf', 
        hugging_face_path='MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF', 
        chat_format="llama-3",
        name='AI Assistant',
        device='cpu',
        max_new_tokens = 1000,
        context_length = 6000
).chat(
    max_new_tokens=512, 
    char_tags=[
        'polite',
        'focused and helpful',
        'expert in programing',
        'obedient'
    ], 
    username='Human',
    temperature=0.8, 
    repetition_penalty=1.3
)