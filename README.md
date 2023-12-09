# flashtorch

A bootstrap LLM loader for CPU/GPU inference.

## Features
- Works for Windows (only CPU tested) and Linux (CPU/GPU)
- Simple and easy to understand, with few objects which handle any case.
- Establishes quick LLM models from scratch ready for application, allows to load and chat in 2 lines of code.
- Can load models to CPU using RAM, or GPU, or both by offloading a specified amount of layers to GPU vram
- Loads models directly from huggingface and store them in local cache
- pytorch and transformer transformer.pipeline compliant - extendable with same keyword arguments from e.g. pipeline
- Has automatic fallbacks for different weight formats (e.g. GGML, GGUF, bin, ..)

## Base Requirements
Assumes drivers, opencl are installed and detects GPU via clinfo, rocminfo, nvidia-smi etc.

## Dependencies

    pip install transformers[torch] accelerate optimum[amd] h5py bitsandbytes

optimum[amd] is used to install the Optimum package with the AMD accelerator.

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