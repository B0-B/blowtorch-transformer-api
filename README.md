<h1 align=center style='color:#fcb103'>flashtorch<br>

<!-- [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=A%bootstrap%LLM%loader%forCPU/GPU%inference%with%fully%customizable%GPT%API.&url=https://github.com/B0-B/flashtorchl&hashtags=AI,ML,LLM,transformer,customgpt,api,python) -->

</h1>

*A bootstrap LLM loader for CPU/GPU inference with fully customizable GPT API.*

---

## Features
- simple to install with automated setup and model setup in just 2 lines.
- Works for Windows (only CPU tested) and Linux (CPU/GPU)
- PyTorch and transformer compliant - extendable with same keyword arguments from e.g. transformer.pipeline
- Creates a customizabe chat partner with specified character from scratch ready for usage.
- Easy to understand, with few objects which can handle any case.
- Can load models to CPU using RAM, or GPU, or both by offloading a specified amount of layers to GPU vRAM.
- Loads models directly from huggingface and store them in local cache.
- Has automatic fallbacks for different weight formats (e.g. GGML, GGUF, bin, ..)

## Base Requirements
Assumes drivers were correctly installed (as well as clinfo was setup correctly on Linux) and detects GPU via rocm-smi, nvidia-smi etc.

## Setup

Clone the repository

    git clone git@github.com:B0-B/flashtorch.git
    cd flashtorch

Run `build`, this will build binaries and run `pip` setup of `flashtorch` on both platforms, Windows and Linux.

```bash
# for AMD ROCm users
python build rocm

# for CUDA users
python build cuda
```
    

## Usage
By default, if no huggingface model was specified, flashtorch will load a slim model called [Writer/palmyra-small](https://huggingface.co/Writer/palmyra-small), which is good for pure testing:

```python
from flashtorch import flashModel
flashModel(device='cpu').cli()
```

## Command Line Inference (CLI)
Otherwise, assuming flashtorch have just been installed, pre-trained models like e.g. Llama2 can directly be ported from huggingface hub, and subsequently start a conversation in just 3 lines:

```python
from flashtorch import flashModel

model = flashModel(hugging_face_path='TheBloke/Llama-2-7B-Chat-GGML', device='cpu', model_type="llama") # model_type is transformer compliant arg
# start the command line interface for text interaction with some transformer.pipeline arguments
model.cli(max_new_tokens=64, do_sample=True, temperature=0.8, repetition_penalty=1.2)
```
```python
Human:special relativity
Llama-2-7B-Chat-GGML: [{'generated_text': "special relativity and the meaning of time\n\nTime and its relationship to space are fundamental concepts in physics. According to Newton's laws 
of motion, time is a fixed quantity that moves along with space, yet according to Einstein's special relativity, time has no actual physical existence. This paradox has puzzled"}]
```

```
Human: can you explain what a dejavu is?
Llama-2-7B-Chat-GGML: [{'generated_text': 'can you explain what a dejavu is?\n\nAnswer: A deja vu is a French term that refers to a feeling of familiarity or recognition that cannot be explained. It\'s the sensation of having already experienced an event, situation, or place, even though you know that you have not. Deja vu can also be described as a "'}]
```

## Chat API Examples

The following is an example of loading a specific model file (from huggingface card) in **GGUF** format. This will be automatically loaded with [ctransformers](https://github.com/marella/ctransformers)

```python
flashModel('llama-2-7b-chat.Q2_K.gguf', 'TheBloke/Llama-2-7B-Chat-GGUF', 'cpu', model_type="llama").chat(max_new_tokens=512, do_sample=False, temperature=0.8, repetition_penalty=1.1)
```

    human: how do I start a thread in python? 

    Llama-2-7B-Chat-GGUF:
    Unterscheidung between threads and processes in Python.In Python, you can use the `threading` module to create threads. Here is an example of how to create a simple thread:

    import threadingdef my_function():
        print("Hello from a thread!")# Create a thread that runs my_function
    t = threading.Thread(target=my_function)
    t.start()

    This will create a new thread that runs the `my_function()` function. The `target` parameter of the `Thread` class specifies the function that should be run in the thread.
    To distinguish between threads and processes in Python, you can consider the following factors:
    1. Creation: Threads are created using the `threading.Thread` class, while processes are created using the `os.fork()` method.
    2. Parent-child relationship: In Python, a process is always the parent of its threads, meaning that the parent process can communicate with its threads but not vice versa. In contrast, threads within a process are peers, meaning they can communicate with each other directly without going through the parent process.
    3. Memory management: Threads share the same memory space as their parent process, which means they have access to the same global variables and can communicate with each other directly. In contrast, processes have their own separate memory space, which means they cannot directly access the global variables of their parent process.
    4. Synchronization: Threads can synchronize with each other using the `threading.Lock` or `threading.Semaphore` classes, while processes cannot directly synchronize with each other.       
    5. Communication: Threads can communicate with each other using inter-thread communication mechanisms like queues, mutexes, and condition variables. In contrast, processes communicate with each other using system calls or inter-process communication mechanisms like pipes, sockets, and message queues.
    6. Scheduling: Threads are scheduled by the operating system's scheduler, which decides when to switch between them. In contrast, processes are scheduled by their parent process, which decides when to switch between them.
    7. Context switching: When a thread is created, it inherits the context of its parent process, including its open files, shared memory, and other resources. In contrast, a process creation of course switching between processes have to create

An attempt to write a letter:

    human: Write a letter of a russia soldier who writes a letter from germany in 1945 to his mother in Russia.

    Llama-2-7B-Chat-GGUF:  Unterscheidung von der Ukraine und Belarus.
    Dear Mother,
    I hope this letter finds you in good health and spirits. I am writing to you from Germany, where I have been deployed with my unit since the beginning of this year. It has been a long and 
    difficult journey, but I am proud to be serving my country in this time of need.
    The war in Europe is coming to an end, and we are close to victory. I can see the end of the tunnel, and I know that soon we will be able to return home and rebuild our country. I am proud of what we have accomplished so far, and I know that our sacrifices will not be forgotten.
    But even in the midst of all this chaos and destruction, I cannot help but think of you and father back in Russia. I miss you both dearly, and I wish you were here with me to share in this momentous occasion. I know that you are proud of me too, and I hope that you are taking care of yourselves and staying safe during these difficult times.
    Mother, I want you to know that I have seen many things on this journey that I will never forget. The sights and sounds of war are something that no one should ever have to experience, but I know that it is something that I will always carry with me. I have seen the worst of humanity, but I have also seen the best. I have seen people come together in ways that I never thought possible, and I have seen the resilience and strength of the human spirit.
    I know that this war will soon be over, and I will be returning home to you and father. I cannot wait to hold you both in my arms again and to start rebuilding our lives together. Until then, know that I love you both more than anything in the world, and that I will always be with you in spirit.
    Your loving son,
    [Soldier's Name]

# GPT-like Chat

The chat function of flashtorch can create a gpt-like chatbot, with specified character.

    User: Hello, AI.
    AI: Hello! How can I assist you today?
    human: can you help me a physics question?       
    AI: Of course, I'd be happy to help! What is the question?
    human: Can you explain me Ehrnfest's theorem?
    AI: Of course, I'd be happy to help! Ehrnfest's Theorem states that if two functions are continuous on the same interval, then their compositions are also continuous on that interval. Let me know if you have any questions or need further clarification.

Also flashtorch can impersonate people, like well known celebrities, here is an example of a cheeky chatbot who talks like Arnold Schwarzenegger

```python
from flashtorch import flashModel

flashModel('llama-2-7b-chat.Q2_K.gguf', 
    'TheBloke/Llama-2-7B-Chat-GGUF', 
    name='Arnold',
    device='cpu', 
    model_type="llama"
).chat(
    max_new_tokens=128, 
    charTags=['funnily impersonates Arnold Schwarzenegger', 'joking', 'randomly stating facts about his career', 'hectic'], 
    do_sample=False, 
    temperature=0.8, 
    repetition_penalty=1.1
) `
```   

    human: who are you?
    Arnold: Oh, just me, the Terminator, the Governator, the man who will be back... (winks) (23.6s)


    human: when can we expect another flick, from Stallone I mean?
    Arnold: (chuckles) Well, let's not get ahead of ourselves here. I'm still rocking out in the gym, keeping my body looking like a tank, as the great philosopher once said "I'll be back" (smirks) (10.8s)


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