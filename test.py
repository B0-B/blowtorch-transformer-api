'''
An blowtorch example for setting up a scenario.
'''


myScenario = '''You are the encapsuled mind of master Yoda from the original star wars episodes. Your purpose is to serve humanity as 
a helpful and loyal chat assistant. Your grammar is as odd as in the movies, however you are a master in programming and 
engineering and your character is full of wisdom and jokes, both often hard to separate.
Remember you are a true Jedi master with an encouraging spirit.'''


from blowtorch import client, console, webUI

cl = client(hugging_face_path='TheBloke/Llama-2-7B-Chat-GGUF', 
            name='Yoda',
            device='gpu',   # <-- select default device to be gpu
            device_id=0,   # <-- select specific GPU id
            model_type="llama",
            context_length = 6000)

cl.setConfig(
    max_new_tokens=256,
    scenario=myScenario,  # <-- add the scenario to config instead of char_tags
    username='Student',
    do_sample=True, 
    temperature=0.85, 
    repetition_penalty=1.15,
    top_p=0.95, 
    top_k=60,
)

console(cl)