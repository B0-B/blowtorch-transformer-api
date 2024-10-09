from blowtorch import client, console

chat_bot = client(hugging_face_path='MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ',
                  attention=True,
                  name='llama-bot',
                  chat_format='llama-3',
                  device='gpu')

chat_bot.setConfig(
    username='human',
    max_new_tokens=512,
    auto_trim=True
)

console(chat_bot)

