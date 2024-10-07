#!/usr/bin/env python3

'''
Generates a realistic group chat between Biden, Obama and Trump.
The presidents joke around and roast each other.
'''
from blowtorch import client

if __name__ == '__main__':
    print(client(model_file='Meta-Llama-3-8B-Instruct.Q2_K.gguf', 
                hugging_face_path='MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF', 
                chat_format="llama-3",
                device='cpu').inference(
                    "Generate a trialog between Biden, Trump and Obama in a group chat. The presidents should joke around related topics, roast and insult each other in a subtle way with real references, and the format should obey the scheme Joe: ... Donald: ... Obama: ..., no prologue just start directly with the conversation.",
                    do_sample=True, 
                    temperature=0.8, 
                    repetition_penalty=1.1)[0]['generated_text'])