from blowtorch import webUI, client

cl = client('llama-2-7b-chat.Q2_K.gguf', 
            'TheBloke/Llama-2-7B-Chat-GGUF', 
            name='AI',
            device='cpu', 
            model_type="llama")

cl.setConfig(
    char_tags=[
        'polite',
        'focused and helpful',
        'expert in programing',
        'obedient'
    ], 
    username='Human',
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