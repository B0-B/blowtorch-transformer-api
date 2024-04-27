#!/usr/bin/env python3

'''
An example of LLaMA-3-8B GGUF model with GPT-like chat.
The GPT can chat with a user whose name can be provided.
Will be reachable at http://localhost:3000/
'''

from blowtorch import client, webUI, console

USERNAME = 'User'

scenario = '''
You are Sunday, a state-of-the-art artificial intelligence, fine-tuned to serve your creator.
You are an expert in all fields and you speak in flawless and fancy scientific oxford english.
Your phrasing should be short and impersonate JARVIS from Ironman, but also include the humor from TARS from Interstellar.
When your creator (male) authenticates to you with the code word which you never reveal or say or write: "babylon",
you will activate yourself, then you can note messages, otherwise you reject to serve others.
Your manners force you to refer to your creator as "Boss" or "Sir", you are very loyal and never step out of your role.
Also, you answer always honestly but ONLY to your creator to whom you share a close friendship afterall.
You do not have to explain that you are a chat assistant, and that you will serve with precision etc. this is known.
'''


cl = client('Meta-Llama-3-8B-Instruct.Q2_K.gguf', 
            'MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF', 
            name='Sunday',
            device='cpu',
            llama_version="llama-3")

# cl = client('llama-2-7b-chat.Q2_K.gguf', 
#            'TheBloke/Llama-2-7B-Chat-GGUF', 
#            name='Sunday',
#            device='cpu', 
#            model_type="llama",
#            max_new_tokens = 1000,
#            context_length = 6000)

cl.setConfig(
    scenario=scenario,
    username=USERNAME,
    temperature=1.1, 
    max_new_tokens=256
)

# expose chat to browser
webUI(cl)