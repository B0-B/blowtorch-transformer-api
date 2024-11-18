<h1 align=center style='color:#fcb103'>blowtorch<br>

<br>

<img src="blowtorch.png" style="text-align:center; width:500px">

<br>
<!-- [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=A%bootstrap%LLM%loader%forCPU/GPU%inference%with%fully%customizable%GPT%API.&url=https://github.com/B0-B/blowtorchl&hashtags=AI,ML,LLM,transformer,customgpt,api,python) -->

</h1>

<p align=center>LLM bootstrap engine for local CPU/GPU inference with fully customizable chat.</p>

---

## 
```python
from blowtorch import client, webUI

USERNAME = 'Steve'

# create state-of-the-art chat bot
myChatClient = client(model_file='Meta-Llama-3-8B-Instruct.Q2_K.gguf', 
                      hugging_face_path='MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF', 
                      chat_format="llama-3",
                      device="cpu")

myChatClient.setConfig(username=USERNAME, max_new_tokens=256, auto_trim=True)

# expose chat in web UI
webUI(myChatClient)
```

## Updates

### Nov 18, 2024
- Fixed the ``client.bench`` method.
  
### Oct 7, 2024
- Added **vLLM** support to ``client`` for accelerated inference - accessible via boolean ``attention`` flag
- Added ``auto_trim`` argument to ``client.chat`` method

### July 3rd, 2024
- Added automated context-length detection and output, so the user is always aware about the current context length.
- **Automated context trimming**: If the current aggregated context is too long for the context length the ``auto_trim=True`` argument (can be called in client init or config) will ensure the most recent context which does not overflow the allowed length. Otherwise users are prone to run into errors like [this](https://github.com/langchain-ai/langchain/issues/3751).
- Added ``cut_unfinished`` argument to client. If enabled all outputs will be truncated to the last fulfilled sentence, unfinished sentences will be cutoff.

### April 27, 2024
-  **Now LLaMA-3 support! See the [example](https://github.com/B0-B/blowtorch-transformer-api/tree/main/examples/llama3_GPT.py).** 🦙🦙🦙 
-  Setup will install latest [xformers](https://github.com/facebookresearch/xformers).

## Features
- Runs on **Windows** (only CPU tested) and **Linux** (CPU/GPU)
- Easy to understand, with single client object for any use-case.
- Supports various llm base modules (transformers, llama.cpp, vllm): *Depending if CPU or GPU device is selected blowtorch auto-loads models in [``transformer.pipeline``](https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/pipelines/__init__.py#L562) or [llama.cpp](https://github.com/ggerganov/llama.cpp) using fallbacks for different weight formats (e.g. GGML, GGUF, bin, ..). Otherwise, if ``attention`` is enabled will use ``vLLM.LLM`` for accelerated inference (if vllm is installed).*
- Simple character or scenarios API for fully customizable chat bots.
- Simple to install and use - a model is setup in just 2 lines.
- Supports various LLaMA prompt embeddings and manages corresponding auto-converts arguments between [``transformer.pipeline``](https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/pipelines/__init__.py#L562), [llama.cpp](https://github.com/ggerganov/llama.cpp) and vLLM.
- Automated model handling, chat and context tracking between multiple converstations.

## Base Requirements   
- Python >=3.10.12
- A system with a CPU (preferably Ryzen) and `>=16GB` RAM
- Assumes drivers were correctly installed and GPU is detectable via rocm-smi, nvidia-smi etc.
- A solid GPT chat requires `>=6GB` of RAM/vRAM depending on device.

## Dependency for performant CPU inference [Default]

This project used to leverage ``ctransformers`` as GGML library for loading GGUF file format. But due to inactivity and incompatibility with new **LLaMA-3** release the backend switched to [``llama-cpp-python``](https://github.com/abetlen/llama-cpp-python) project. This python API provides c-bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp). 

**blowtorch** uses ``llama.cpp`` in parallel to classic transformers for more and better onboarding options with CPU focus and quantized models.

|**library**|**version**|
|-|-|
|transformers|4.43.2|
|llama-cpp-python|latest|
|accelerate|0.30.0|
|h5py|3.9.0|
|psutil|latest|
|optimum|latest|
|auto-gptq|0.7.1|
|tokenizers|0.19.1|
|*ctransformers*|**deprecated**|


## Docs

<!-- SETUP -->
<details>
<summary style="font-size:2rem">Setup</summary>

---

### PIP Wheel 

Will automatically install latest pre-built release

    pip install https://b0-b.github.io/blowtorch-transformer-api/dist/blowtorch-1.3.1-py3-none-any.whl

### Manual Installation
Clone the repository

    git clone https://github.com/B0-B/blowtorch-transformer-api.git
    cd blowtorch-transformer-api

Install the provided wheel distribution via python script

    python install.py 

or with ``pip`` package manager

    pip install ./dist/blowtorch-1.3.1-py3-none-any.whl 

Alternatively, if a hardware specific build is needed just build from source using automated script.

    python rebuild.py

``Note:`` This will create a new package wheel in the ``./dist`` branch with your current settings. To install the build run **python install.py**. To build and directly install subsequently run

    python rebuild.py && python install.py

### GPU & BLAS Backends for llama.cpp 
blowtorch distinguishes between model formats suited for CPU or GPU. If GPU is selected it will out-of-the-box attempt to load it with ``transformers`` (if suited) which leverages the default torch BLAS backend. If you intend to load a GGUF model on GPU however, blowtorch will try to load it with **llama.cpp**. For this re-build ``llama-cpp-python`` with the corresponding BLAS (linear algebra instruction) backend. You can find the full build instructions in [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python) or the summarized commands below

```bash
# CPU acceleration on MacOS/Linux
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
# CPU acceleration on Windows
$env:CMAKE_ARGS = "-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

# ROCm hipBLAS on MacOS/Linux
CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
# ROCm hipBLAS on Windows
$env:CMAKE_ARGS = "-DLLAMA_HIPBLAS=on"
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

# CUDA on Linux
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
# CUDA on MacOS
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
# CUDA on Windows
$env:CMAKE_ARGS = "-DLLAMA_CUDA=on" 
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```
    


</details>  


<!-- USAGE -->
<details>
<summary style="font-size:2rem">Usage</summary>

---

## Getting-Started

Blowtorch builds on the client analogy where the model and necessary parameters are held by one object ``blowtorch.client``. The client is the main object which will allow to do all manipulations and settings for our model, like LLM transformer parameters, a name and character etc.

By default, if no huggingface model was specified, blowtorch will load a slim model called [Writer/palmyra-small](https://huggingface.co/Writer/palmyra-small), which is good for pure testing and can be considered the simplest test

```python
from blowtorch import client
client(device='cpu')
```

Generally, LLMs are designed to predict the next word in a sequence. Loading an LLM and generating from inputs like a started sentence, it will try to finish the sentence. For a chat-like experience, blowtorch exploits and tracks the context and initializes the chat with attributes (and character), which allows the AI to track the context and reason accordingly.

First, to download and run an arbitrary huggingface model

```python
cl = client(hugging_face_path='MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ', 
            name='GPT',
            device='gpu', # <-- select GPU as device
            device_id=0,  # <-- optionally select the GPU id
            model_type="llama-3",
            trust_remote_code=False,
            revision="main")
```
also, you can give your client a name, model_type (should match the current model), and it's possible to pre-define some transformers kwargs, but those can be overriden by ``cli`` or ``chat`` method kwargs.

For a gpt-chat in the console one should either use the ``chat`` method

```python
cl.chat(
    max_new_tokens=128, 
    char_tags=[
        'polite',
        'focused and helpful',
        'expert in programing',
        'obedient'
    ], 
    username='Human',  
    temperature=0.8, 
    repetition_penalty=1.1)
```

Alternatively, the ``setConfig`` method allows to do the pre-configuration analogously

```python
# it is recommended to first set the config
cl.setConfig(
    char_tags=[
        'carring comrade',
        'polite',
        'focused and helpful',
        'expert in programing',
        'obedient'
    ], 
    username='Human',
    temperature=0.8, 
    repetition_penalty=1.1
)

cl.chat() # no arguments needed anymore
```

## Accelerated Inference
Blowtorch API can access flash-attn. backend via vLLM (requires vLLM install with python module) for an accelerated chat on GPU. Simply enable the attention flag in the client to access the feature.

```python
chat_bot = client(hugging_face_path='MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ',
                  attention=True,       # setting attention to true
                  quantization='gptq',  # set correct quantization (needed when attention is enabled)
                  name='llama-bot',
                  chat_format='llama-3',
                  device='gpu')
```

## Exposers for your Chat

The two main ways to expose your chat are

 - **console** - which runs in the console (terminal) of your current runtime. *Alias for client.chat.*
 - **webUI** - which starts a webserver with hosted UI in the browser

which can be imported in your python project

```python
from blowtorch import console, webUI
```

As shown in this snippet, ``blowtorch.console`` object can be used as an alias for `blowtorch.chat` method but demands setting a config apriori. The chat arguments can also be pre-loaded (often useful) with the ``setConfig`` method. Then all other methods (like chat) or exposing objects require no arguments anymore. Note, variables ``do_sample, temperature, repetition_penalty`` are additional ``transformer`` kwargs, that will be accepted as well. 

Once the configuration of a client is setup, it may be exposed via a **web server** for a better GUI **(for more info see web UI section)**

```python
cl.setConfig(
    char_tags=[
        'carring comrade',
        'polite',
        'focused and helpful',
        'expert in programing',
        'obedient'
    ], 
    username='Human',
    temperature=0.8, 
    repetition_penalty=1.1
)

# expose web service
from blowtorch import webUI
webUI(cl)
```



</details>



<!-- GPT CHAT  -->
<details>
<summary style="font-size:2rem">Custom Character</summary>

## Character Tags & Scenarios

### Character Tags
The chat function of blowtorch can create a custom flavored (or guided) chat without any training with a specified character or a whole impersonation. Here is an example of a cheeky chatbot who talks like Arnold Schwarzenegger

```python
from blowtorch import client

client('llama-2-7b-chat.Q2_K.gguf', 
    'TheBloke/Llama-2-7B-Chat-GGUF', 
    name='Arnold',
    device='cpu', 
    model_type="llama"
).chat(
    max_new_tokens=128, 
    char_tags=['funnily impersonates Arnold Schwarzenegger', 'joking', 'randomly stating facts about his career', 'hectic'], 
    temperature=0.8, 
    repetition_penalty=1.1
) `
```   

    human: who are you?
    Arnold: Oh, just me, the Terminator, the Governator, the man who will be back... (winks) (23.6s)

    human: when can we expect another flick, from Stallone I mean?
    Arnold: (chuckles) Well, let's not get ahead of ourselves here. 
    I'm still rocking out in the gym, keeping my body looking like a tank, 
    as the great philosopher once said "I'll be back" (smirks) (10.8s)

    human: Well I think youhave a dad bod! 
    Arnold: (laughs) Dad bod?! (chuckles) Kid, I'm 74 years old, and I'm feeling better than ever! Don't believe these young whippersnappers who say I've gone soft. I'm still the baddest of them all! (flexes)
    (15.2s)

    human: Can you give me a step-by-step guide to improve my abs please?
    Arnold: (chuckles) Ah, kid, you want to know how to get those washboard abs like mine? Let me tell you something, it takes more than just a few crunches to become the Terminator. You gotta train hard, eat right, and never give up! (pounds chest) (11.8s)

also we can play a game of **guess who**

    human: Let's play guess who
    Arnold: Ah, a game of wit and charm! *adjusts sunglasses* I am always ready to put on a show for my dear friend here! *winks* 

    human: Ok I am thinking of a character, and you are allowed to ask yes or no questions until you make the right pick, got it?
    Arnold: *nods* Of course, my dear! *grinning* Ask away, my dear! *wags eyebrows*
    human: now you ask me a question
    Arnold: *thoughtfully* Hmmm...is this person a historical figure? *winks*
    human: yes
    Arnold: *excitedly* Oh, that narrows it down to like...a billion people! *chuckles* But seriously, how did you know I was thinking of someone from history? *raises an eyebrow*

    human: Well the person I am thinking of is definitely an historic figure, just continue questioning you are on a good track!
    Arnold: *smirks* Ah, I see! *leaning in* So, does this person have any superpowers? *winks*

    human: No
    Arnold: *disappointed* Oh noooo...I was hoping for a challenge! *pouts* But fear not, my dear! *adjusts sunglasses* I shall still try my best to solve this puzzle! *determined look* So, tell me more about this person...is he/she from ancient times? *tips sunglasses*

    human: Yes, the person is from ancient times!
    Arnold: *excitedly* Oh boy, this is getting interesting! *nods* So, this person lived over 2000 years ago? *asks innocently* And what else can you tell me about them? *curious expression*
    human: Yes!

### Scenarios
Besides the ``char_tags`` to give your chat bot attributes or shape his character a bit,
the ``setConfig`` method provides a more in-depth initialization option called ``scenario`` to give users more freedom to create their personalized main frame. An example of a scenario where a film scene is depicted for a cosplay between the user and the AI

```python 
myScenario = '''This is the scene in the movie "heat", where you, Robert Deniro (with caricaturized behaviour), and me, Al Pacino, are meeting face-to-face for the first time in a diner.'''

cl = client('llama-2-7b-chat.Q2_K.gguf', 
            'TheBloke/Llama-2-7B-Chat-GGUF', 
            name='Deniro',
            device='cpu', 
            model_type="llama",
            context_length = 6000)

cl.setConfig(
    max_new_tokens=128,
    scenario=myScenario,  # <-- add the scenario to config instead of char_tags
    username='Pacino',
    temperature=0.85, 
    repetition_penalty=1.15,
    top_p=0.95, 
    top_k=60,
)
```
</details>



<!-- WEB UI  -->
<details>
<summary style="font-size:2rem">Web UI</summary>

---

The API comes with a web interface implementation for better I/O. It serves all the necessary needs however should be considered PoC at this stage to demonstrate how to create applications by using blowtorch under the hood.
Here is an example screenshot running exposed on local host

<p align="center"><img width=600 src='./demo.PNG' ></p>

`webUI` is a ``client``-wrapper which will expose your client, once it's configured for production (e.g. using the setConfig method) as such

```python
cl.setConfig(
    char_tags=[
        'carring comrade',
        'polite',
        'focused and helpful',
        'expert in programing',
        'obedient'
    ], 
    username='Human',
    temperature=0.8, 
    repetition_penalty=1.1
)

from blowtorch import webUI
webUI(cl, port=3000)
```

**Note:** Every TCP connection, i.e. browser window, tab will initiliaze a new session ID which is passed to the server who keeps track of different conversations and distinguishes them across the local network.

</details>




<!-- BENCHMARKS -->
<details>
<summary style="font-size:2rem">Benchmarks</summary>

---

`blowtorch` comes with a built-in benchmark feature. Assuming a configured client, loaded with a model of choice, the bench method can be called for performance metrics and memory usage. Note that for proper measurement and better estimate, the benchmark performs a 512 token generation which can take around a minute.

```python
cl = client('llama-2-7b-chat.Q2_K.gguf', 
            'TheBloke/Llama-2-7B-Chat-GGUF', 
            name='AI',
            device='cpu', 
            model_type="llama",
            context_length = 6000)

cl.bench()
```

    info: start benchmark ...

    -------- benchmark results --------
    Device: AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD
    RAM Usage: 3.9 gb
    vRAM Usage: 0 b
    Max. Token Window: 512
    Tokens Generated: 519
    Bytes Generated: 1959 bytes
    Token Rate: 6.701 tokens/s
    Data Rate: 25.294 bytes/s
    Bit Rate: 202.352 bit/s
    TPOT: 149.231 ms/token
    Total Gen. Time: 77.448 s

The results show that the total RAM consumption (of the total python process) takes around $3.9GB$.

</details>

