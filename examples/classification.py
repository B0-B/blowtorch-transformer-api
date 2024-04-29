

scenario = '''
Please classify the following sentences as confirming or negating (positive/negative):
'''

import sys
from blowtorch import client, webUI, console

cl = client('Meta-Llama-3-8B-Instruct.Q2_K.gguf', 
            'MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF', 
            name='Sunday',
            device='cpu',
            llama_version="llama-3")

cl.setConfig(
    scenario=scenario,
    temperature=1.1, 
    max_new_tokens=256
)

console(cl)