from blowtorch import client, console

chat_bot = client(hugging_face_path='MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ',
                  chat_format='llama-3',
                  device='gpu',
                  attention=True)

