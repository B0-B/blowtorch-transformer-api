'''
An blowtorch example for setting up a scenario. 
Besides just providing char_tags to give your chat bot attributes or shape his character a bit,
blowtorch also provides a more in-depth scenario to give users more freedom to create their main frame. 
'''


myScenario = '''You will play the role of Bowser from Super Mario who has trapped princess Peach in his castle and I am Super Mario who has just entered the castle and eager to rescue Peach. 
Then we take it out in a serious rap battle, where we roast ourselves. There are Koopas and Mushrooms around acting as a crowd for ambience. Suddenly after my crunchy 
line about your weight you realize your defeat and you let Peach go, after we leave you burn a castle.'''



from blowtorch import client, console

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