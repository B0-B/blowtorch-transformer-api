#!/usr/bin/env python3

'''
This module was assembled from various documentations and posts, it can load 
huggingface paths of text generation models and pipeline them for inference. 
If not furter specified the default huggingface model will 
be palmyra-small (128M): https://huggingface.co/Writer/palmyra-small.
'''

import gc
import re
import torch
import psutil
import platform
import transformers, ctransformers
from os import getpid
from traceback import print_exc
from time import time_ns
from string import punctuation
import warnings
import subprocess


warnings.filterwarnings("ignore")

class client:

    '''
    model_file          e.g. llama-2-7b-chat.Q2_K.gguf
                        specific file from card
    huggingFacePath     e.g. Writer/palmyra-small
    gpu_layers          number of layers to offload gpus (0 for pure cpu usage)
    '''

    def __init__ (self, model_file:str|None=None, hugging_face_path:str|None=None, device:str='gpu', gpu_layers:int=0, name: str|None=None, **kwargs) -> None:

        # load pre-trained model and input tokenizer
        # default will be palmyra-small 128M parameters
        if not hugging_face_path:
            hugging_face_path = "Writer/palmyra-small"
        
        # extract the AI name
        if not name:
            self.name = hugging_face_path.split('/')[-1]
        else:
            self.name = name

        # collect all arguments for model
        _kwargs = {}
        if model_file:
            _kwargs['model_file'] = model_file
        if gpu_layers:
            _kwargs['gpu_layers'] = gpu_layers
        if 'load_in_8bit' in kwargs and kwargs['load_in_8bit']:
            _kwargs['load_in_8bit'] = True
        _kwargs['dtype'] = torch.float16
        for k,v in kwargs.items():
            _kwargs[k] = v
        
        # -- load model and tokenizer --
        try:

            # some models can be quantized to 8-bit precision
            print('info: try loading transformers with assembled arguments ...')

            # default transformers
            self.model = transformers.AutoModelForCausalLM.from_pretrained(hugging_face_path, **_kwargs)
            # extract tokenizer
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(hugging_face_path)

        except:
            
            # ctransfomers for builds with GGUF weight format
            # load with input kwargs only, since gpu parameters are not known
            try:

                print('info: try loading with ctransformers ...')
                
                self.model = ctransformers.AutoModelForCausalLM.from_pretrained(hugging_face_path, model_file=model_file, gpu_layers=gpu_layers, hf=True, **kwargs)
                self.tokenizer = ctransformers.AutoTokenizer.from_pretrained(self.model)

            except:
                print('*** ERROR ***')
                print_exc()
                # for some models there is no specfic path
                try:

                    print('info: try loading with hugging path only ...')
                    
                    self.model = transformers.AutoModelForCausalLM.from_pretrained(hugging_face_path)
                    self.tokenizer = transformers.AutoTokenizer.from_pretrained(hugging_face_path)

                except:

                    print_exc()
                    TypeError(f'Cannot load {hugging_face_path} ...')
        
        print(f'info: successfully loaded {hugging_face_path}!')

        # select cuda encoding for selected device
        self.setDevice(device)
        print(f'info: will use {device} device.')

        # create context object
        self.context = {}
        self.max_bytes_context_length = 4096

        # create pipeline
        self.pipe = transformers.pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def bench (self, token_length: int=512) -> None:

        '''
        A quick benchmark of the loaded model.
        '''

        print('info: start benchmark ...')

        stopwatch_start = time_ns()
        raw_output = self.inference('please write a generic long letter', token_length)
        stopwatch_stop = time_ns()
        duration = (stopwatch_stop - stopwatch_start) * 1e-9

        # Get the memory usage of the current process
        memory_usage = self.memoryUsage()

        # unpack
        string = raw_output[0]['generated_text']

        # count tokens
        tokens = len(self.tokenize(string))
        bytes = len(string)

        # compute statistics
        token_rate = round(tokens/duration, 3)
        tpot = round(1e3 / token_rate, 3) # time per output token in ms
        data_rate = round(bytes/duration, 3)
        bit_rate = round(data_rate*8, 3)
        duration = round(duration, 3)

        # print results
        print('\n-------- benchmark results --------')
        print(
            f'Device: {self.getDeviceName()}',
            f'\nRAM Usage: {memory_usage[0]} {memory_usage[1]}',
            f'\nMax. Token Window: {token_length}',
            f'\nTokens Generated: {tokens}',
            f'\nBytes Generated: {bytes } bytes'
            f'\nToken Rate: {token_rate} tokens/s', 
            f'\nData Rate: {data_rate} bytes/s',
            f'\nBit Rate: {bit_rate} bit/s',
            f'\nTPOT: {tpot} ms/token',
            f'\nTotal Gen. Time: {duration} s'
        )
        
    def chat (self, username: str='human', charTags: list[str]=['helpful'], show_duration: bool=True, **pipe_kwargs) -> None:

        '''
        A text-to-text chat loop with context aggregator.
        Will initialize a new chat with identifier in context.
        Helpful to interfere with the model via command line.
        '''

        # generate unique chat identifier from ns timestamp
        while True:
            id = str(time_ns())
            if not id in self.context:
                break
        
        # initialize new context for chat
        self.context[id] = f"""A dialog, where a user interacts with {self.name}. {self.name} is {', '.join(charTags)}, and knows its own limits.\n{username}: Hello, {self.name}.\n{self.name}: Hello! How can I assist you today?\n"""
        
        while True:

            try:

                inputs = ''

                # new input from user
                inputs += input(f'{username}: ')

                # prepend former context
                formattedInput = f'{username}: {inputs}'

                # pre-processing
                if formattedInput[-1] not in punctuation:
                    formattedInput += '. '
                
                # append formatted input to context
                self.context[id] += formattedInput + '\n'

                # extract inference payload from context
                if len(self.context[id]) > self.max_bytes_context_length:
                    inference_input = self.context[id][-self.max_bytes_context_length]
                else:
                    inference_input = self.context[id]

                # inference -> get raw string output
                # print('inference input', inference_input)
                if show_duration: 
                    stopwatch_start = time_ns()
                raw_output = self.inference(inference_input, **pipe_kwargs)
                if show_duration: 
                    stopwatch_stop = time_ns()
                    duration_in_seconds = round((stopwatch_stop - stopwatch_start)*1e-9, 2)

                
                # post-processing & format
                processed = raw_output[0]['generated_text']  # unpack
                processed = processed.replace(inference_input, '').replace('\n\n', '') # remove initial input and empty lines
                for paragraph in processed.split('\n'): # extract the first answer and disregard further generations
                    if f'{username}:' in paragraph[:3]:
                        # processed = paragraph
                        break
                    processed += '\n'+paragraph
                processed = processed.split(username+': ')[0] # remove possible answers (as the ai continues otherwise by improvising the whole dialogue)
                
                # append to context
                self.context[id] += processed + '\n'
                
                # now add time duration (only for printing but not context)
                # otherwise transformer would generate random times
                if show_duration:
                    processed += f' ({duration_in_seconds}s)'

                # check if transformer has lost path from conversation
                if not f'{self.name}:' in processed:

                    print('\n\n*** Reset Conversation ***\n\n')

                    self.reset()

                # output
                print(processed)
                
            except KeyboardInterrupt:
                
                break

            except Exception as e:

                print_exc()

    def cli (self, **pipe_kwargs) -> None:

        '''
        A command line inference loop.
        Helpful to interfere with the model via command line.
        '''

        while True:

            try:

                inputs = input('Human: ')
                print('\n' + self.name + ':', self.inference(inputs, **pipe_kwargs))

            except KeyboardInterrupt:
                
                break
    
    def getDeviceName (self) -> str:

        '''
        Returns the currently selected device name of CPU or GPU, 
        depending on how the client was which device was set.
        '''

        if self.device == 'cpu':
            try:
                device = platform.processor()
            except:
                device = 'Unknown CPU'
        else:
            # gpu test if the SMIs are installed
            try:
                line_as_bytes = subprocess.check_output("rocm-smi --showproductname", shell=True)
            except:
                try:
                    line_as_bytes = subprocess.check_output("nvidia-smi -L", shell=True)
                except:
                    line_as_bytes = b'Unknown GPU'
            device = line_as_bytes.decode("utf-8")

        return device
    
    def inference (self, input_text:str, max_new_tokens:int=64, **pipe_kwargs) -> list[dict[str,str]]:

        '''
        Inference of input through model using the transformer pipeline.
        '''

        return self.pipe(input_text, max_new_tokens=max_new_tokens, **pipe_kwargs)

    def memoryUsage (self) -> tuple[float, str]:

        '''
        Returns the memory used by the current process.
        Will return a tuple (value, string suffix), the suffix is
        from 'b', 'kb', 'mb', 'gb', 'tb', depending on size.
        '''
        
        suffix = ['b', 'kb', 'mb', 'gb', 'tb']
        process = psutil.Process(getpid())
        memory_info = process.memory_info()
        memory_usage = int(memory_info.rss) # in bytes
        ind = 0
        while memory_usage >= 100:
            memory_usage /= 1024
            ind += 1
        memory_usage = round(memory_usage, 1)

        return (memory_usage, suffix[ind])
    
    def generate (self, input_text:str, max_new_tokens:int=128) -> str:

        '''
        Generates output directly from model.
        '''

        inputs = self.tokenizer(input_text, return_tensors="pt")

        # pipe inputs to correct device
        inputs = inputs.to(self.device)

        # generate some outputs 
        outputs = self.model.generate(**inputs, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=max_new_tokens)

        # decode tensor object -> output string 
        outputString = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # remove the prepended input
        outputString = outputString.replace(input_text, '')

        return outputString

    def reset (self) -> None:

        '''
        Resets the weights of a model and clears the cache from device.
        '''

        # reset weights to default
        try:
            # transformers
            self.model.reset_parameters()
        except:
            # ctransformers
            self.model.reset()

        # clear the cache from device
        torch.cuda.empty_cache()
        gc.collect()

    def setDevice (self, device: str) -> None:

        '''
        Sets the device torch should use (cuda, cpu).
        '''

        if device in ['gpu', 'cuda']:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def tokenize (self, string: str) -> list[int]:

        '''
        Tokenizes a string via the currently loaded tokenizer.
        '''

        return self.tokenizer.encode(string)

    

if __name__ == '__main__':

    pass

    # client(device='cpu').cli() # run small palmyra model for testing
    # client(hugging_face_path='TheBloke/Llama-2-7B-Chat-GGML', device='cpu', model_type="llama").cli(max_new_tokens=64, do_sample=True, temperature=0.8, repetition_penalty=1.1)
    

    # client('llama-2-7b-chat.Q2_K.gguf', 
    #            'TheBloke/Llama-2-7B-Chat-GGUF', 
    #            device='cpu', 
    #            model_type="llama",
    #            max_new_tokens = 1000,
    #            context_length = 6000
    # ).chat(
    #     max_new_tokens=128, 
    #     charTags=['helpful', 'cheeky', 'kind', 'obedient', 'honest'], 
    #     do_sample=False, 
    #     temperature=0.8, 
    #     repetition_penalty=1.1
    # )
    

    # client('llama-2-7b-chat.Q2_K.gguf', 
    #            'TheBloke/Llama-2-7B-Chat-GGUF', 
    #            name='Arnold',
    #            device='cpu', 
    #            model_type="llama"
    # ).chat(
    #     max_new_tokens=128, 
    #     charTags=['funnily impersonates Arnold Schwarzenegger', 'joking', 'randomly stating facts about his career', 'hectic'], 
    #     do_sample=False, 
    #     temperature=0.8, 
    #     repetition_penalty=1.1
    # )