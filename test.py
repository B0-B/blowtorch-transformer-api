from blowtorch import client, webUI

cl = client('llama-2-7b-chat.Q2_K.gguf', 
            'TheBloke/Llama-2-7B-Chat-GGUF', 
            name='AI',
            device='cpu', 
            model_type="llama",
            max_new_tokens = 1000,
            context_length = 6000)

cl.setConfig(
    max_new_tokens=128, 
    char_tags=[
        'imressionates Lisa Su', 
        'speaks like a CEO from AMD', 
        'versed in semi-conductor technology',
        'kind', 
        'eloquent',
        'genuine'
    ], 
    username='User',
    do_sample=False, 
    temperature=0.8, 
    repetition_penalty=1.1
)

cl.chat()
# webUI(cl)