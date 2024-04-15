#!/usr/bin/env python3

'''
Generates a realistic group chat between Biden, Obama and Trump.
The presidents joke around and roast each other.
'''
from blowtorch import client

if __name__ == '__main__':
    print(client('llama-2-7b-chat.Q2_K.gguf', 'TheBloke/Llama-2-7B-Chat-GGUF', 
        device='cpu', model_type="llama").inference(
                "Generate a trialog between Biden, Trump and Obama in a group chat. The presidents should joke around related topics, roast and insult each other in a subtle way with real references, and the format should obey the scheme Joe: ... Donald: ... Obama: ..., no prologue just start directly with the conversation.",
                
                do_sample=True, 
                temperature=0.8, 
                repetition_penalty=1.1)[0]['generated_text'])