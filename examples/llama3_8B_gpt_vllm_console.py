#!/usr/bin/env python3

'''
A GPT-like example of LLama-3-8B chat accelerated with vLLM's flash-attn. backend.
'''
from blowtorch import client, console

chat_bot = client(hugging_face_path='MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ',
                  attention=True, # setting attention to true
                  name='llama-bot',
                  chat_format='llama-3',
                  device='gpu')

chat_bot.setConfig(
    username='human',
    max_new_tokens=512,
    auto_trim=True
)

console(chat_bot)

