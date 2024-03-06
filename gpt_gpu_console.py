#!/usr/bin/env python3

'''
Creates GPT chat in web browser.
Will be reachable at http://localhost:3000/
'''


from blowtorch import client, console

cl = client(hugging_face_path='TheBloke/Llama-2-7b-Chat-GPTQ', 
            name='GPT',
            device='gpu', 
            model_type="llama",
            trust_remote_code=False,
            revision="main",
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
    do_sample=True, 
    top_p=0.95, 
    top_k=40,
    temperature=0.85, 
    repetition_penalty=1.15
)

console(cl)