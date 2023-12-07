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
Human:can you explain what a dejavu is?
Llama-2-7B-Chat-GGML: [{'generated_text': 'can you explain what a dejavu is?\n\nAnswer: A deja vu is a French term that refers to a feeling of familiarity or recognition that cannot be explained. It\'s the sensation of having already experienced an event, situation, or place, even though you know that you have not. Deja vu can also be described as a "'}]
```
