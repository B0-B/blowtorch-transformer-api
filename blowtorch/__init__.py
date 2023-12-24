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
from traceback import print_exc, format_exc
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



# __________ logging __________
# colors for logging
__colors__ = {
    'b': '\033[94m',
    'c': '\033[96m',
    'g': '\033[92m',
    'y': '\033[93m',
    'r': '\033[91m',
    'nc': '\033[0m'
}

class client:

    '''
    model_file          e.g. llama-2-7b-chat.Q2_K.gguf
                        specific file from card
    huggingFacePath     e.g. Writer/palmyra-small
    device              'cpu' or 'gpu'
    gpu_layers          number of layers to offload gpus (0 for pure cpu usage)
    name                the name of the client
    verbose             if true will disable all warnings
    silent              if true will diable all console output
    **twargs            custom trasformer key word arguments
    '''

    def __init__ (self, model_file:str|None=None, hugging_face_path:str|None=None, device:str='gpu', gpu_layers:int=0, 
                  name: str|None=None, verbose: bool=False, silent: bool=False, **twargs) -> None:

        # stdout console logging
        self.silent = silent
        if not verbose:
            # filter general warnings in stdout
            warnings.filterwarnings("ignore")
            # disable specifically transfomer warnings
            # this will suppress fp8, fp16, flash attention etc. 
            transformers.logging.set_verbosity_warning()

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
        _twargs = {}
        if model_file:
            _twargs['model_file'] = model_file
        if gpu_layers:
            _twargs['gpu_layers'] = gpu_layers
        if 'load_in_8bit' in twargs and twargs['load_in_8bit']:
            _twargs['load_in_8bit'] = True
        _twargs['dtype'] = torch.float16
        for k,v in twargs.items():
            _twargs[k] = v
        
        # -- load model and tokenizer --
        try:

            # some models can be quantized to 8-bit precision
            self.log('try loading transformers with assembled arguments ...', label='⚙️')

            # default transformers
            self.model = transformers.AutoModelForCausalLM.from_pretrained(hugging_face_path, **_twargs)
            # extract tokenizer
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(hugging_face_path)

        except:
            
            # ctransfomers for builds with GGUF weight format
            # load with input twargs only, since gpu parameters are not known
            try:

                self.log('try loading with ctransformers ...', label='⚙️')
                
                self.model = ctransformers.AutoModelForCausalLM.from_pretrained(hugging_face_path, model_file=model_file, gpu_layers=gpu_layers, hf=True, **twargs)
                self.tokenizer = ctransformers.AutoTokenizer.from_pretrained(self.model)

            except:
                print('*** ERROR ***')
                print_exc()
                # for some models there is no specfic path
                try:
                    
                    self.log('try loading with hugging path only ...', label='⚙️')
                    
                    self.model = transformers.AutoModelForCausalLM.from_pretrained(hugging_face_path)
                    self.tokenizer = transformers.AutoTokenizer.from_pretrained(hugging_face_path)

                except:

                    self.log(f'Cannot load {hugging_face_path} ...', label='🛑')
                    ModelLoadingError(format_exc())
        
        self.log(f'successfully loaded {hugging_face_path}!', label='✅')

        # select cuda encoding for selected device
        self.setDevice(device)
        self.log(f'will use {device} device.', label='⚠️')

        # create context object
        self.context = {}
        self.max_bytes_context_length = 4096

        # twargs config, which can be pre-set before calling inference, cli or chat
        # use setConfig to define inference twargs
        self.config = None

        # create pipeline
        self.pipe = transformers.pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def bench (self, token_length: int=512) -> None:

        '''
        A quick benchmark of the loaded model.
        '''

        self.log('start benchmark ...', label='⏱️')

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
        
    def chat (self, username: str='human', char_tags: list[str]=['helpful'], show_duration: bool=True, **pipe_twargs) -> None:

        '''
        A text-to-text chat loop with context aggregator.
        Will initialize a new chat with identifier in context.
        Helpful to interfere with the model via command line.

        If setConfig was called priorly, pipe_twargs will be overriden.
        If setConfig includes the method twargs it will override as well.
        '''

        # clarify twargs
        if self.config:
            conf = self.config
            # override standard twargs with existing 
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
            # override pipe twargs with left twargs in config
            pipe_twargs.update(conf)

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
                raw_output = self.inference(inference_input, **pipe_twargs)
                if show_duration: 
                    stopwatch_stop = time_ns()
                    duration_in_seconds = round((stopwatch_stop - stopwatch_start)*1e-9, 2)

                # post-processing & format
                processed = raw_output[0]['generated_text']  # unpack
                processed = processed.replace(inference_input, '').replace('\n\n', '') # remove initial input and empty lines
                user_tags = [f'{username}:'.lower(), f'{username};'.lower()]
                ai_tags = [f'{self.name}:'.lower(), f'{self.name};'.lower()]
                processed_rohling = ''
                for paragraph in processed.split('\n'): # extract the first answer and disregard further generations
                    # check if the paragraph refers to AI's output
                    # otherwise if it's a random user generation terminate
                    head = paragraph[:2*len(user_tags[0])].lower()
                    if user_tags[0] in head or user_tags[1] in head or ai_tags[0] in head or ai_tags[1] in head:
                        break
                    processed_rohling += '\n'+paragraph
                # remove possible answers (as the ai continues otherwise by improvising the whole dialogue)
                # override processed variable with rohling
                processed = processed_rohling.split(username+': ')[0] 

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

    def cli (self, **pipe_twargs) -> None:

        '''
        A command line inference loop.
        Helpful to interfere with the model via command line.
        '''

        while True:

            try:

                inputs = input('Human: ')
                print('\n' + self.name + ':', self.inference(inputs, **pipe_twargs))

            except KeyboardInterrupt:
                
                break
    
    def contextInference (self, input_text:str, sessionId: int=0, username: str='human', char_tags: list[str]=['helpful'], **pipe_twargs):

        '''
        Mimicks the chat method, but returns the output.
        Used for webUI inference, will behave the same as chat method.

        If setConfig was call priorly twargs, the pre-defined twargs will be overriden.
        '''

        # clarify twargs by merging with config (if enabled)
        if self.config:
            conf = self.config
            # override standard twargs with existing 
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
            # override pipe twargs with left twargs in config
            pipe_twargs.update(conf)

        # initialize new context by registering sessionId in context object
        if not sessionId in self.context:
            self.log(f'Start new conversation {sessionId}', label='🗯️')
            self.context[sessionId] = f"""A dialog, where a user interacts with {self.name}. {self.name} is {', '.join(char_tags)}, and knows its own limits.\n{username}: Hello, {self.name}.\n{self.name}: Hello! How can I assist you today?\n"""

        # format input
        formattedInput = f'{username}: {input_text}'

        # pre-processing
        if formattedInput[-1] not in punctuation:
            formattedInput += '. '
        
        # firstly, slice the context feed by the max. allowed size
        if len(self.context[sessionId]) > self.max_bytes_context_length:
            inference_input = self.context[sessionId][-self.max_bytes_context_length]
        else:
            inference_input = self.context[sessionId]
        
        # then append formatted input to context and inference input
        self.context[sessionId] += formattedInput + '\n'
        inference_input += formattedInput + '\n'

        # inference -> get raw string output
        # print('inference input', inference_input)
        raw_output = self.inference(inference_input, **pipe_twargs)

        # post-processing & format
        processed = raw_output[0]['generated_text']  # unpack
        processed = processed.replace(inference_input, '').replace('\n\n', '') # remove initial input and empty lines
        user_tag = f'{username}:'.lower()
        # aiTag = f'{self.name}:'.lower()
        processed_rohling = ''
        for paragraph in processed.split('\n'): # extract the first answer and disregard further generations
            # check if the paragraph refers to AI's output
            # otherwise if it's a random user generation terminate
            if user_tag in paragraph[:2*len(user_tag)].lower():
                break
            processed_rohling += '\n'+paragraph
        # remove possible answers (as the ai continues otherwise by improvising the whole dialogue)
        # override processed variable with rohling
        processed = processed_rohling.split(username+': ')[0] 
        
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
    
    def inference (self, input_text:str, **pipe_twargs) -> list[dict[str,str]]:

        '''
        Inference of input through model using the transformer pipeline.
        '''
        print('pipe twargs:', pipe_twargs)
        return self.pipe(input_text, **pipe_twargs)

    def log (self, *stdout: any, label='info', color='c') -> None:

        '''
        Logs args to console.
        '''

        if self.silent:
            return

        header = f"{__colors__[color]}{label.upper()}{__colors__['nc']} "
        print(header, *stdout)

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
    
    def reset (self) -> None:

        '''
        Resets the weights of a model and clears the cache from device.
        '''

        # reset weights to default
        try:
            # transformers
            self.model.reset_parameters()
        except:
            try:
                # ctransformers
                self.model.reset()
            except:
                self.log('cannot reset model:', format_exc(), label='🛑')

        # clear the cache from device
        torch.cuda.empty_cache()
        gc.collect()
    
    def setConfig (self, **twargs) -> None:

        '''
        An alternative settings method for transformer twargs.
        The provided twargs will be forwarded to inference and chat.
        Usage:
        _client = client(...)
        _client.setConfig(twargs...)
        _client.chat() # no twargs needed anymore
        '''

        self.config = twargs

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

        self.__client__.log('data received:', data, label='📩')

        # check if package is consistent
        if not data['sessionId']:

            err = 'No sessionId provided!'
            response['errors'].append(err)
            self.__client__.log(err, label='🛑')

        elif not data['message']:
            
            err = 'No message provided in request!'
            response['errors'].append(err)
            self.__client__.log(err, label='🛑')

        elif not (data['maxNewTokens'] and type(data['maxNewTokens']) is int and data['maxNewTokens'] > 64):
            
            err = 'maxNewTokens must be >=64!'
            response['errors'].append(err)
            self.__client__.log(err, label='🛑')
        
        else:

            # extract session id so the client can locate the context
            sessionId = data['sessionId'] 
            
            # do inference
            message = data['message']
            maxNewTokens = data['maxNewTokens']
            output = self.__client__.contextInference(
                message, 
                sessionId=sessionId,
                max_new_tokens=maxNewTokens
            )

            self.__client__.log('send:', output, label='📨')

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

        '''
        Starts the TCP server with handler class.
        '''

        with socketserver.TCPServer((self.host, self.port), handler) as httpd:
            print(f'📡 serving blowtorch UI server at http://{self.host}:{self.port}')
            httpd.serve_forever()


class ModelLoadingError (Exception):

    pass