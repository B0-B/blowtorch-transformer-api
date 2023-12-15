#!/usr/bin/env python3

'''
This module was assembled from various documentations and posts, it can load 
huggingface paths of text generation models and pipeline them for inference. 
If not furter specified the default huggingface model will 
be palmyra-small (128M): https://huggingface.co/Writer/palmyra-small.
'''

import gc
import re
import json
from time import time_ns
from string import punctuation
import warnings

# system
import psutil
import subprocess
import platform
from traceback import print_exc
from pathlib import Path, PosixPath
from os import getpid, chdir, getcwd

# ML
import torch
import transformers, ctransformers

# server
import _socket
import socketserver
from http.server import SimpleHTTPRequestHandler


__root__ = Path(__file__).parent
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

        # kwargs config, which can be pre-set before calling inference, cli or chat
        # use setConfig to define inference kwargs
        self.config = None

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
        memory_usage = self.ramUsage() # memory occupied by the process in total
        vram_usage = self.vramUsage() # memory allocated b torch on gpu

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
            f'\nvRAM Usage: {vram_usage[0]} {vram_usage[1]}',
            f'\nMax. Token Window: {token_length}',
            f'\nTokens Generated: {tokens}',
            f'\nBytes Generated: {bytes } bytes'
            f'\nToken Rate: {token_rate} tokens/s', 
            f'\nData Rate: {data_rate} bytes/s',
            f'\nBit Rate: {bit_rate} bit/s',
            f'\nTPOT: {tpot} ms/token',
            f'\nTotal Gen. Time: {duration} s'
        )
        
    def chat (self, username: str='human', char_tags: list[str]=['helpful'], show_duration: bool=True, **pipe_kwargs) -> None:

        '''
        A text-to-text chat loop with context aggregator.
        Will initialize a new chat with identifier in context.
        Helpful to interfere with the model via command line.

        If setConfig was called priorly, pipe_kwargs will be overriden.
        If setConfig includes the method kwargs it will override as well.
        '''

        # clarify kwargs
        if self.config:
            conf = self.config
            # override standard kwargs with existing 
            # config value and remove from config
            if 'username' in conf:
                username = conf['username']
                conf.pop('username')
            if 'char_tags' in conf:
                char_tags = conf['char_tags']
                conf.pop('char_tags')
            if 'show_duration' in conf:
                show_duration = conf['show_duration']
                conf.pop('show_duration')
            # override pipe kwargs with left kwargs in config
            pipe_kwargs = conf

        # generate unique chat identifier from ns timestamp
        while True:
            id = str(time_ns())
            if not id in self.context:
                break
        
        # initialize new context for chat
        self.context[id] = f"""A dialog, where a user interacts with {self.name}. {self.name} is {', '.join(char_tags)}, and knows its own limits.\n{username}: Hello, {self.name}.\n{self.name}: Hello! How can I assist you today?\n"""
        
        while True:

            try:

                inputs = ''

                # new input from user
                inputs += input(f'{username}: ')

                # format input
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

                    self.context[id] = f"""A dialog, where a user interacts with {self.name}. {self.name} is {', '.join(char_tags)}, and knows its own limits.\n{username}: Hello, {self.name}.\n{self.name}: Hello! How can I assist you today?\n"""

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
    
    def contextInference (self, input_text:str, sessionId: int=0, username: str='human', char_tags: list[str]=['helpful'], **pipe_kwargs):

        '''
        Mimicks the chat method, but returns the output.
        Used for webUI inference, will behave the same as chat method.

        If setConfig was call priorly kwargs, the pre-defined kwargs will be overriden.
        '''

        # clarify kwargs
        if self.config:
            conf = self.config
            # override standard kwargs with existing 
            # config value and remove from config
            if 'username' in conf:
                username = conf['username']
                conf.pop('username')
            if 'char_tags' in conf:
                char_tags = conf['char_tags']
                conf.pop('char_tags')
            # override pipe kwargs with left kwargs in config
            pipe_kwargs = conf

        # initialize new context by registering sessionId in context object
        if not sessionId in self.context:
            self.context[sessionId] = f"""A dialog, where a user interacts with {self.name}. {self.name} is {', '.join(char_tags)}, and knows its own limits.\n{username}: Hello, {self.name}.\n{self.name}: Hello! How can I assist you today?\n"""

        # format input
        formattedInput = f'{username}: {input_text}'

        # pre-processing
        if formattedInput[-1] not in punctuation:
            formattedInput += '. '
        
        # append formatted input to context
        self.context[sessionId] += formattedInput + '\n'

        # extract inference payload from context
        if len(self.context[sessionId]) > self.max_bytes_context_length:
            inference_input = self.context[sessionId][-self.max_bytes_context_length]
        else:
            inference_input = self.context[sessionId]

        # inference -> get raw string output
        # print('inference input', inference_input)
        raw_output = self.inference(inference_input, **pipe_kwargs)

        # post-processing & format
        processed = raw_output[0]['generated_text']  # unpack
        processed = processed.replace(inference_input, '').replace('\n\n', '') # remove initial input and empty lines
        for paragraph in processed.split('\n'): # extract the first answer and disregard further generations
            if f'{username}:' in paragraph[:3]:
                # processed = paragraph
                break
            processed += '\n'+paragraph
        processed = processed.split(username+': ')[0] # remove possible answers (as the ai continues otherwise by improvising the whole dialogue)
        
        # check if transformer has lost path from conversation
        if not f'{self.name}:' in processed:
            print('\n\n*** Reset Conversation ***\n\n')
            processed = f"{self.name}: Let's start a new chat!"
            self.reset()
            self.context[sessionId] = f"""A dialog, where a user interacts with {self.name}. {self.name} is {', '.join(char_tags)}, and knows its own limits.\n{username}: Hello, {self.name}.\n{self.name}: Hello! How can I assist you today?\n"""
            return processed

        # append to context
        self.context[sessionId] += processed + '\n'

        return processed

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

    def ramUsage (self) -> tuple[float, str]:

        '''
        Returns the memory used by the current process.
        Will return a tuple (value, string suffix), the suffix is
        from 'b', 'kb', 'mb', 'gb', 'tb', depending on size.
        '''
        
        suffix = ['b', 'kb', 'mb', 'gb', 'tb']

        # measure RAM usage
        process = psutil.Process(getpid())
        memory_info = process.memory_info()
        memory_usage = int(memory_info.rss) # in bytes

        # select correct suffix
        ind = 0
        while memory_usage >= 100:
            memory_usage /= 1024
            ind += 1
        memory_usage = round(memory_usage, 1)

        return (memory_usage, suffix[ind])
    
    def vramUsage (self) -> tuple[float, str]:

        '''
        Returns the current vRAM usage.
        '''

        suffix = ['b', 'kb', 'mb', 'gb', 'tb']

        try:
            # get the vRAM allocated by torch
            vram_usage = torch.cuda.memory_allocated(0)
        except:
            # set to zero if torch is not compiled for gpu
            vram_usage = 0
        
        # select correct suffix
        ind = 0
        while vram_usage >= 100:
            vram_usage /= 1024
            ind += 1
        vram_usage = round(vram_usage, 1)

        return (vram_usage, suffix[ind])

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

    def setConfig (self, **kwargs) -> None:

        '''
        An alternative settings method for transformer kwargs.
        The provided kwargs will be forwarded to inference and chat.
        Usage:
        _client = client(...)
        _client.setConfig(kwargs...)
        _client.chat() # no kwargs needed anymore
        '''

        self.config = kwargs

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

class handler (SimpleHTTPRequestHandler):

    '''
    blowtorch http handler for web serving and TCP interface.
    '''

    __client__: client
    
    def do_GET(self):
        
        '''
        Load static web page.
        '''

        return SimpleHTTPRequestHandler.do_GET(self)
    
    def do_POST(self):

        '''
        Chat API implementation.
        '''

        response = {'data': {}, 'errors': []}

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        # unpack the request
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))

        print('data received:', data)

        # check if package is consistent
        if not data['sessionId']:

            err = 'No sessionId provided!'
            response['errors'].append(err)
            print(err)

        elif not data['message']:
            
            err = 'No message provided in request!'
            response['errors'].append(err)
            print(err)

        elif not (data['maxNewTokens'] and type(data['maxNewTokens']) is int and data['maxNewTokens'] > 64):
            
            err = 'maxNewTokens must be >=64!'
            print(err)
            response['errors'].append(err)
        
        else:

            # extract session id so the client can locate the context
            sessionId = data['sessionId'] 
            
            # do inference
            message = data['message']
            maxNewTokens = data['maxNewTokens']
            output = self.__client__.contextInference(
                sessionId,
                message, 
                max_new_tokens=maxNewTokens,
            )

            # add output to data
            response['data']['message'] = output

        # prepare response
        encoded = json.dumps(response).encode('utf-8')

        # send
        self.wfile.write(encoded)

class webUI ():

    '''
    Spawns a blowtorch server with a web UI.
    _client:    client instance
    dir:        path to static directory
    host:       host name
    port:       TCP port
    '''

    def __init__(self, _client: client, dir: str|Path=__root__.joinpath('static/'), host: str='localhost', port: int=3000) -> None:
        
        self.host = host
        self.port = port
        
        # override global client for the handler 
        self.client = _client
        handler.__client__ = self.client

        # change current wdir for socketserver
        origin = getcwd()
        chdir(dir)
        try:
            self.startServer()
        except:
            print_exc()
        finally:
            chdir(origin)

    def startServer (self):

        with socketserver.TCPServer((self.host, self.port), handler) as httpd:
            print(f'serving blowtorch web UI at http://{self.host}:{self.port}')
            httpd.serve_forever()


if __name__ == '__main__':

    webUI('')

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
    #     char_tags=['helpful', 'cheeky', 'kind', 'obedient', 'honest'], 
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
    #     char_tags=['funnily impersonates Arnold Schwarzenegger', 'joking', 'randomly stating facts about his career', 'hectic'], 
    #     do_sample=False, 
    #     temperature=0.8, 
    #     repetition_penalty=1.1
    # )