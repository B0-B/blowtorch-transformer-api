from blowtorch import webUI, client

cl = client('llama-2-7b-chat.Q2_K.gguf', 'TheBloke/Llama-2-7B-Chat-GGUF', device='cpu', model_type="llama")

webUI()