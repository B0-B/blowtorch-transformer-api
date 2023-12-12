from blowtorch import client
import subprocess

# try:
#     line_as_bytes = subprocess.check_output("rocm-smi --showproductname", shell=True)
# except:
#     line_as_bytes = subprocess.check_output("nvidia-smi -L", shell=True)
# return line_as_bytes.decode("utf-8")
client(hugging_face_path='TheBloke/Llama-2-7B-Chat-GGML', device='cpu', model_type="llama").bench()

#!/usr/bin/env python3

if __name__ == '__main__' and False:
    print(client('llama-2-7b-chat.Q2_K.gguf', 'TheBloke/Llama-2-7B-Chat-GGUF', 
        device='cpu', model_type="llama").inference(
                "Generate a trialog between Biden, Trump and Obama. The presidents should joke around random topics and insult each other in a subtle way with real references, The format should obey the scheme Joe: ... Donald: ... Obama: ..., no prologue just start directly with the conversation.",
                512, 
                do_sample=True, 
                temperature=0.8, 
                repetition_penalty=1.1)[0]['generated_text'])