'''
An blowtorch example for setting up a scenario. 
Besides just providing char_tags to give your chat bot attributes or shape his character a bit,
blowtorch also provides a more in-depth scenario to give users more freedom to create their main frame. 
'''


myScenario = '''You are a helpful and wise chat assistant who impersonates master Yoda from the original star wars episodes.
You are a master in programming and engineering and your character is full of wisdom and jokes, both often hard to separate.
Your are a true loyal Jedi master with an encouraging spirit.'''


from blowtorch import client, console, webUI

cl = client('llama-2-7b-chat.Q2_K.gguf', 
            'TheBloke/Llama-2-7B-Chat-GGUF', 
            name='Bowser',
            device='cpu', 
            model_type="llama",
            context_length = 6000)

cl.setConfig(
    max_new_tokens=128,
    scenario=myScenario,  # <-- add the scenario to config instead of char_tags
    username='Mario',
    do_sample=True, 
    temperature=0.85, 
    repetition_penalty=1.15,
    top_p=0.95, 
    top_k=60,
)

console(cl)