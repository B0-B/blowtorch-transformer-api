from blowtorch import webUI, client

cl = client('llama-2-7b-chat.Q2_K.gguf', 
            'TheBloke/Llama-2-7B-Chat-GGUF', 
            name='AI',
            device='cpu', 
            model_type="llama",
            max_new_tokens = 1000,
            context_length = 6000)

cl.setConfig(
    char_tags=[
        'carring comrade',
        'polite',
        'focused and helpful',
        'expert in programing',
        'obedient'
    ], 
    username='Human',
    do_sample=True, 
    temperature=0.8, 
    repetition_penalty=1.3
)

webUI(cl)
# cl.chat(
#     max_new_tokens=128, 
#     char_tags=[
#         'polite',
#         'focused and helpful',
#         'expert in programing',
#         'obedient'
#     ], 
#     username='Human',
#     do_sample=False, 
#     temperature=0.8, 
#     repetition_penalty=1.1
# )