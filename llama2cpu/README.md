# Setup

1. Download llama.cpp build `llama-b1617-bin-win-avx2-x64.zip` from the official release repository and decompress the build into a folder e.g. "llama-build".
2. Download the Llama models in GGUF format from [TheBloke/Llama-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF) repository. This format is compliant with llama.cpp. For example download the model `llama-2-7b-chat.ggmlv3.q2_K.bin`

3. Run the model
    ```python 
   .\llama-build\main -t 8 -m llama-2-7b-chat.Q2_K.gguf --color -c 2048 --temp 0.7 --repeat_penalty 1.1 -n -1 -p 'Can you tell me something about yourself?' 
   ```