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

    def __init__ (self, model_file:str|None=None, hugging_face_path:str|None=None, device:str='gpu', device_id:None|int=None,
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

        # select device
        self.device = 'cpu'
        self.setDevice(device)
        self.log(f'will use {device} device.', label='⚠️')

        # collect all arguments for model
        #   removed model_file as deprecated in transformers but not in ctransformers
        _twargs = {}
        for k,v in twargs.items():
            _twargs[k] = v
        if 'load_in_8bit' in twargs and twargs['load_in_8bit']:
            _twargs['load_in_8bit'] = True
        
        
        # -- load model and tokenizer --
        self.model = None
        self.tokenizer = None
        model_loaded = self.loadModel(model_file, hugging_face_path, device, device_id, **twargs)
        if not model_loaded:
            exit()
        else:
            self.log(f'successfully loaded {hugging_face_path}!', label='✅')

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
        raw_output = self.inference('please write a generic long letter', max_new_tokens=token_length)
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
        
    def chat (self, username: str='human', char_tags: list[str]=['helpful'], scenario: str=None, show_duration: bool=True, **pipe_twargs) -> None:

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
            # config value and remove from config since they are not transformer kwargs
            if 'username' in conf:
                username = conf['username']
                conf.pop('username')
            if 'char_tags' in conf:
                char_tags = conf['char_tags']
                conf.pop('char_tags')
            if 'show_duration' in conf:
                show_duration = conf['show_duration']
                conf.pop('show_duration')
            if 'scenario' in conf:
                scenario = conf['scenario']
                conf.pop('scenario')
            # override pipe twargs with left twargs in config
            pipe_twargs.update(conf)

        # generate unique chat identifier from ns timestamp
        while True:
            id = str(time_ns())
            if not id in self.context:
                break
        
        # initialize new context for chat by providing
        # - either a scenario as a string
        # - character tags provided as a list of strings
        self.setContext(id, username, char_tags, scenario)

        while True:

            try:

                # new input from user
                new_input = input(f'{username}: ')
                
                # start duration measurement
                if show_duration: 
                    stopwatch_start = time_ns()
                
                # inference -> get raw string output
                # forward the new input through context pipeline
                processed_output = self.contextInference(new_input, sessionId=id, username=username, 
                                                         char_tags=char_tags, scenario=scenario, **pipe_twargs)
                
                # stop watch
                if show_duration: 
                    stopwatch_stop = time_ns()
                    duration_in_seconds = round((stopwatch_stop - stopwatch_start)*1e-9, 2)

                # append to context
                # self.context[id] += processed + '\n'
                formatted_output = f'{self.name}: {processed_output}'

                # now add time duration (only for printing but not context)
                # otherwise transformer would generate random times
                if show_duration:
                    formatted_output += f' ({duration_in_seconds}s)'

                # output
                print(formatted_output)
                
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
    
    def contextInference (self, input_text:str, sessionId: int=0, username: str='human', char_tags: list[str]=['helpful'], scenario: None|str=None, **pipe_twargs):

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
                conf.pop('show_duration')
            if 'scenario' in conf:
                scenario = conf['scenario']
                conf.pop('scenario')

            # override pipe twargs with left twargs in config
            pipe_twargs.update(conf)

        # initialize new context by registering sessionId in context object
        if not sessionId in self.context:
            self.log(f'Start new conversation {sessionId}', label='🗯️')
            self.setContext(sessionId, username, char_tags, scenario)

        

        # formatted input
        # formattedInput = f'{username}: {input_text}'
            
        # ---- pre-processing ----    
        # load current context from history (extract first element i.e. SYS-tag)
        # https://github.com/facebookresearch/llama/issues/484#issuecomment-1649286345
        # https://huggingface.co/blog/llama2#how-to-prompt-llama-2
        messages = [self.context[sessionId][0]]
        do_strip = False
        for user_input, response in self.context[sessionId][1:]:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            messages.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')

        message = input_text.strip() if do_strip else input_text
        messages.append(f'{message} [/INST]')

        # construct the final prompt
        prompt = ''.join(messages)

        # inference -> get raw string output and response
        raw_output = self.inference(prompt, **pipe_twargs)
        response = raw_output[0]['generated_text']
        
        # post-processing & format
        processed_output = self.postProcess(prompt, raw_output[0]['generated_text'], username)
        # processed_tagged = f'{processed}</s><s>'

        # append q&a tuple to context
        self.context[sessionId].append((input_text, processed_output))

        return processed_output

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

    def loadModel (self, model_file:str|None=None, hugging_face_path:str|None=None, device:str='gpu', device_id:None|int=None, **twargs) -> bool:

        '''
        Loads model onto selected device, up to a specific GPU.

        device          device gpu/cuda, cpu
        device_id       None (default), 0, 1, ...
        twargs          Transformer arguments
        model_file      only for ctransformers

        Returns boolean to indicate success.
        '''

        # ==== GPU approach ====
        if device.lower() in ['gpu', 'cuda']:

            # select the proper device
            if device_id:
                cuda_arg = f'cuda:{device_id}'
            else:
                cuda_arg = 'cuda'
            
            self.log(f'try loading {hugging_face_path} onto GPU', label='⚙️')

            # try loading with auto device map and assembled arguments
            try:

                self.log('try loading transformers with auto device map and provided arguments ...', label='⚙️')

                # default transformers
                self.model = transformers.AutoModelForCausalLM.from_pretrained(hugging_face_path, device_map=cuda_arg, **twargs)

                # extract tokenizer
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(hugging_face_path, use_fast=True)
                self.log(f'Successfully loaded {hugging_face_path} on GPU', label='✅')
                return True

            except:

                print_exc()
            
            # try fixed revision and dont trust remote code 
            # this approach is suitable for Llama-2-7b-Chat-GPTQ
            try:

                self.log(f'try loading {hugging_face_path} with fixed revision and mistrust remote code ...', label='⚙️')

                # override kwargs
                twargs['revision'] = 'main'
                twargs['trust_remote_code'] = False

                # default transformers
                self.model = transformers.AutoModelForCausalLM.from_pretrained(hugging_face_path, device_map=cuda_arg, **twargs)

                # extract tokenizer
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(hugging_face_path, use_fast=True)
                self.log(f'Successfully loaded {hugging_face_path} on GPU', label='✅')
                return True

            except:

                print_exc()
            
            # try loading without arguments
            try:
                self.log(f'try loading {hugging_face_path} blank (no args) ...', label='⚙️')
                transformers.AutoModelForCausalLM.from_pretrained(hugging_face_path, device_map=cuda_arg)
                self.log(f'Model was loaded but may behave differently since no arguments were provied. The arguments were dropped to enable the onloading to {device.upper()}.', label='⚠️')
                return True
            except:
                print_exc()

            self.log(f'failed loading onto GPU', label='🛑')


        # ==== CPU approach ====
        try:

            self.log('try loading transformers on CPU using provided arguments ...', label='⚙️')

            # default transformers
            self.model = transformers.AutoModelForCausalLM.from_pretrained(hugging_face_path, device_map="cpu", **twargs)

            # extract tokenizer
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(hugging_face_path, use_fast=True)
            self.log(f'Successfully loaded {hugging_face_path} on CPU', label='✅')
            return True

        except:

            print_exc()

        # ctransfomers for builds with GGUF weight format
        # load with input twargs only, since gpu parameters are not known
        try:

            self.log(f'try loading {hugging_face_path} with ctransformers ...', label='⚙️')
            
            self.model = ctransformers.AutoModelForCausalLM.from_pretrained(hugging_face_path, model_file=model_file, hf=True, **twargs)
            self.tokenizer = ctransformers.AutoTokenizer.from_pretrained(self.model)
            self.log(f'Successfully loaded {hugging_face_path} with ctransformers on CPU', label='✅')
            return True
        
        except:

            print_exc()
        
        self.log(f'failed loading {hugging_face_path} onto CPU as well.', label='🛑')
        
        return False

    def log (self, *stdout: any, label='info', color='c') -> None:

        '''
        Logs args to console.
        '''

        if self.silent:
            return

        header = f"{__colors__[color]}{label.upper()}{__colors__['nc']} "
        print(header, *stdout)

    def postProcess (self, _input:str, _output:str, username:str) -> str:

        '''
        Post-processing method which takes raw _input and _output from LLM and returns a post-processed output.
        This is only raw message and string processing, no transformer tags are added. 
        '''

        # define processed output
        processed = _output  

        # remove initial input and empty lines
        processed = processed.replace(_input, '').replace('\n\n', '') 

        # sort out valid paragraphs
        user_tag = f'{username}:'.lower()
        processed_rohling = ''
        for paragraph in processed.split('\n'): # extract the first answer and disregard further generations
            # check if the paragraph refers to AI's output
            # otherwise if it's a random user generation terminate
            if user_tag in paragraph.lower():
                break
            processed_rohling += '\n' + paragraph
        
        # override with processed paragraphs
        processed = processed_rohling
        
        # remove possible answers (as the ai continues otherwise by improvising the whole dialogue)
        # override processed variable with rohling
        if user_tag in processed_rohling.lower():
            processed = processed.split(username+': ')[0] 
        
        return processed

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
    
    def setContext (self, id: str, username: str, char_tags: list[str]=['helpful'], scenario: None|str=None) -> None:

        '''
        Sets context provided by character tags (list) or scenario (str).
        which will be initialized in the current chat id.
        '''
        
        if scenario:
            ctx = scenario
        else:
            ctx = f"""This is a dialog, where a user interacts with {self.name}.\n {self.name} is {', '.join(char_tags)}, and is aware of his limits. \n{username}: Hello, who are you?\n{self.name}: Hello! I am **{self.name}** How can I assist you today?"""
        
        # initialize new chat according to https://huggingface.co/blog/llama2#how-to-prompt-llama-2
        self.context[id] = [f'<s>[INST] <<SYS>>\n{ctx}\n<</SYS>>\n\n']

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
        torch.set_default_device(self.device)

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


class console ():

    '''
    Class alias for chat method.
    Will expose the chat in the console.
    '''

    def __init__(self, _client:client) -> None:
        _client.chat()


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
    Spawns a blowtorch server with exposed chat in a web UI.

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