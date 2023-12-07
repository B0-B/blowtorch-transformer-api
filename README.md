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

    pip install transformers[torch] accelerate optimum h5py bitsandbytes

## Usage
By default, if no huggingface model was specified, flashtorch will load a slim model called [Writer/palmyra-small](https://huggingface.co/Writer/palmyra-small), which is good for pure testing:

```python
from flashtorch import flashModel
flashModel(device='cpu').cli()
```
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

Another example with **GGUF** format on [ctransformers](https://github.com/marella/ctransformers)

```python
flashModel('llama-2-7b-chat.Q2_K.gguf', 'TheBloke/Llama-2-7B-Chat-GGUF', 'cpu', model_type="llama").cli(max_new_tokens=64, do_sample=True, temperature=0.8, repetition_penalty=1.1)
```

```
Human: how can i write a hello world program in python?

Llama-2-7B-Chat-GGUF: [{'generated_text': 'how can i write a hello world program in python?\n вересня 17, 2022 at 1:48 pm\nA "Hello, World!" program in Python is a simple program that prints the message "Hello, World!" to the screen. Here are the steps to write a "Hello, World!" program in Python:\n1. Open a text editor or an Integrated Development Environment (IDE) and create a new file.\n2. Write the following code in the file:\n```\nprint("Hello, World!")\n```\n3. Save the file with a ".py" extension (e.g., "hello_world.py").\n4. Open a terminal or command prompt and navigate to the directory where you saved the file.\n5. Type "python" followed by the name of your file (e.g., "python hello_world.py").\n6. Press Enter to run the program.\nHere\'s what will happen when you run the program wished you were here; it will print "Hello, World!" to the screen, just like any other Python program!\nNote that the "print()" function is used to print messages to the screen in Python. The parentheses after the "print()" function indicate that this is a statement'}]
```

```
Human: can you write a sad letter of a british soldier to his mother from germany in 1942?

Llama-2-7B-Chat-GGUF: [{'generated_text': 'can you write a sad letter of a british soldier to his mother from germany in 1942?   \n Hinweis: I am not an historian, so please excuse any historical inaccuracies in my response.  \n\nDear Mother,\n\nI hope this letter finds you in good health and spirits despite the turmoil that surrounds us. I am writing to you today from the 
midst of war, here in Germany where I have been stationed with my regiment for the past few months. It has been a grueling and unrelenting experience, but I know that you, as my loving mother, will want to hear about it directly from me.\nThe days are dark and damp, and the nights colder than ever before. We are constantly on the move, never staying in one place for too long. The enemy is closing in from all sides, and we are fighting for every inch of ground. I fear for my safety every day, and I cannot shake off the feeling of constant danger that looms over me. But I know that I am not alone in this fight, and that gives me some measure of comfort. My comrades are brave men, and together we are determined to see this through to the end.\nBut despite the bravery and determination of my fellow soldiers, the'}]
```
