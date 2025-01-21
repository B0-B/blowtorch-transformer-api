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
from time import time_ns, perf_counter_ns
from string import punctuation
import warnings

# system
import os
import psutil
import subprocess
import platform
from traceback import print_exc, format_exc
from pathlib import Path, PosixPath
from os import getpid, chdir, getcwd

# ML
import torch
import transformers
from llama_cpp import Llama
from llama_cpp.llama_tokenizer import LlamaHFTokenizer
try:
    from vllm import LLM, SamplingParams
    __ATTN__ = True
except ImportError:
    __ATTN__ = False
    Warning('[⚠️] No vllm installation found, cannot use attention.')


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

# ---- clients ----
class BaseClient:

    '''
    A huggingface-compliant AI client for running (large-) language models. 
    Supports automated loading of models and information handling, character 
    and prompt feeding API, context- and session tracking, as well as bench options.

    To interface the client instance, it can be passed to an expose object 
    such as console() or webUI().
    '''

    def __init__ (self, 
                  model_file: str|None=None, 
                  hugging_face_path: str|None=None, 
                  attention: bool=False,
                  chat_format: str='llama-3', 
                  device: str='gpu', 
                  device_id: int=0,
                  name: str|None=None, 
                  verbose: bool=True, 
                  silent: bool=False, 
                  **twargs) -> None:
        
        '''
        [Parameters]

        model_file          e.g. llama-2-7b-chat.Q2_K.gguf
                            specific file from huggingface card
        huggingFacePath     e.g. Writer/palmyra-small
        device              'cpu' or 'gpu'
        gpu_layers          number of layers to offload gpus (0 for pure cpu usage)
        name                the name of the client
        verbose             if true will disable all warnings
        silent              if true will disable all console output
        **twargs            custom trasformer key word arguments
        '''

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
        self.hugging_face_path = hugging_face_path
        
        # extract the AI name
        if not name:
            self.name = hugging_face_path.split('/')[-1]
        else:
            self.name = name

        # collect all arguments for model
        #   removed model_file as deprecated in transformers but not in ctransformers
        _twargs = {}
        for k,v in twargs.items():
            _twargs[k] = v
        if 'load_in_8bit' in twargs and twargs['load_in_8bit']:
            _twargs['load_in_8bit'] = True

        # try to detect llama version
        self.chat_format = chat_format
        for temp in ['llama-2', 'llama-3']:
            if temp in hugging_face_path.lower() and chat_format != temp:
                self.log(f'recognized that "{hugging_face_path}" is a {temp} model, but provided the chat_format is "{chat_format}" which could cause problems while prompting!', label='⚠️') if verbose else None
                break
            elif temp in hugging_face_path.lower() and chat_format == temp:
                break

        # use vllm.LLM model as LLM base module corpus if attention is enabled:
        # - for accelerated GPU inference
        if attention:

            # check if vllm is really installed
            if not __ATTN__:
                ImportError('Cannot use attention: No vllm installation found. Please install vllm.')

            self.log('attention activated -> routing to vllm ...', label='⚠️') if verbose else None
            self.llm_base_module = 'vllm'

            # unfortunately it isn't possible to pin the LLM object to a specific cuda device:
            # https://github.com/vllm-project/vllm/issues/3750
            # will therefore use the environment workaround
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
            self.device = 'gpu'
            self.device_id = device_id
            self.selectDevice(device, self.device_id)

            # load the model and tokenizer
            self.model = LLM(hugging_face_path, tokenizer=hugging_face_path, **twargs)
            self.tokenizer = self.model.get_tokenizer()
            self.pipe = self.model.generate

        # otherwise check which base module to use:
        # - transformers module for GPU
        # - llama.cpp module for CPU
        else:

            # select device and device_id and initialize with method
            self.device = 'cpu'
            self.device_id = device_id
            self.selectDevice(device, self.device_id)
            self.log(f'will use {device} device.', label='⚠️') if verbose else None            
            
            # init default library/module used, 
            # client.loadModel will change that accordingly.
            self.llm_base_module = 'transformers'

            # -- load model and tokenizer and instantiate pipeline --

            # load model and tokenizer
            self.model = None
            self.tokenizer = None
            model_loaded = self.loadModel(model_file, hugging_face_path, device, device_id, **twargs)
            if not model_loaded:
                exit()
            else:
                self.log(f'successfully loaded {hugging_face_path}!', label='✅') if verbose else None
                
            # create pipeline based on base module
            if self.llm_base_module == 'transformers':
                self.pipe = transformers.pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
            elif self.llm_base_module == 'llama.cpp':
                # for llama.cpp the model can be used as pipe
                # see: https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#high-level-api
                self.pipe = self.model

        # create context object
        self.context = {}
        # extract the maximum allowed context length
        # this will get helpful for context trimming.
        try:
            if self.llm_base_module == 'transformers':
                self.context_length = int(self.model.max_seq_length) 
            elif self.llm_base_module == 'llama.cpp':
                self.context_length = int(self.model.n_ctx()) 
            elif self.llm_base_module == 'vllm':
                self.context_length = self.model.config.n_ctx
            else:
                ValueError()
            self.log(f'Context length detected: {self.context_length}', label='⚠️') if verbose else None
        except:
            self.context_length = 512
            self.log(f'cannot extract context_length, - set default to 512.', label='⚠️') if verbose else None
        
        # twargs config, which can be pre-set before calling inference, cli or chat
        # use setConfig to define inference twargs
        self.config = None

    def chat (self, 
              username: str='human', 
              char_tags: list[str]=['helpful'], 
              scenario: str=None, 
              show_duration: bool=True, 
              **pipe_twargs) -> None:

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

        # generate unique chat identifier 
        id = self.newSessionId()
        
        # initialize new context for chat by providing
        # - either a scenario as a string
        # - character tags provided as a list of strings
        self.newConversation(id, username, char_tags, scenario)

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
                print(formatted_output, end='\n')
                
            except KeyboardInterrupt:
                
                break

            except Exception as e:

                print_exc()

    def contextInference (self, input_text: str, sessionId: int=0, username: str='human', char_tags: list[str]=['helpful'],
                          scenario: None|str=None, cut_unfinished:bool=False, auto_trim: bool=False, **pipe_twargs) -> str:

        '''
        Inference with context tracking.
        Supports LLaMA-2 and LLaMA-3 prompting formats.
        Auto formatting of prompts and post processing included.
        For benchmarks better use client.inference.

        [Parameters]
        input_text              Prompt string.
        sessionId               The current conversation identifier.
                                Different sessionId will have a different context
        username                Name/role of the user.
        char_tags               List with string attributes to characterize the assistant 
                                e.g. ['helpful', 'obedient', 'technically versed'].
                                Will only work if scenario is None.
        scenario                A string scenario where everything is specified in a fluent
                                text string, this will makes char_tags obsolete.
        cut_unfinished          If enabled, will remove unfinished sentences at the end.
        auto_trim               Will automatically trim the context to allowed context_length,
                                if it gets too long for propagation.
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
            if 'scenario' in conf:
                scenario = conf['scenario']
                conf.pop('scenario')
            if 'cut_unfinished' in conf:
                cut_unfinished = conf['cut_unfinished']
                conf.pop('cut_unfinished')
            if 'auto_trim' in conf:
                auto_trim = conf['auto_trim']
                conf.pop('auto_trim')

            # override pipe twargs with left twargs in config
            pipe_twargs.update(conf)
        
        # convert twargs to llama.cpp if cpu is used
        pipe_twargs = self.__convert_twargs__(pipe_twargs)

        # (fuse): initialize new context by registering sessionId 
        # in context object if none exists already.
        if not sessionId in self.context:
            self.newConversation(sessionId, username, char_tags, scenario)

        # Determine the recent context which will be propagated.
        recent_context = self.context[sessionId]
        # trim the context
        if auto_trim:
            
            # trimming index - start from whole conversation
            threshold = 0.9 # [between 0 and 1]
            n = len(self.context[sessionId]) - 1
            verbose = True
            while True:

                trimmed_context = self.context[sessionId][:1] + self.context[sessionId][1:][-n:]
                concat_context = trimmed_context[0]

                for p, a in trimmed_context[1:]:
                    concat_context += p + ' ' + a
                
                total_tokens = len(self.tokenize(concat_context))

                if total_tokens > threshold * self.context_length:
                    self.log('trimming context to allowed length ...', label='⚙️') if verbose else None
                    verbose = False
                    n -= 1 # trim further
                else:
                    recent_context = trimmed_context
                    break

        # Gather a formatted conversation with formatted system_prompt.
        formatted_conversation = [recent_context[0]]
        do_strip = False
        
        # Reconstruct conversation with correct prompting format
        for user_input, response in recent_context[1:]:

            user_input = user_input.strip() if do_strip else user_input
            do_strip = True

            formatted_prompt = self.__format_prompt__(user_input, header=username, response=response)
            formatted_conversation.append( formatted_prompt )

        # Append current message derived from input.
        message = input_text.strip() if do_strip else input_text
        formatted_message = self.__format_prompt__(message, username)
        formatted_conversation.append(formatted_message)

        # Merge formatted conversation to final prompt
        formatted_prompt = ''.join(formatted_conversation)

        # -> inference
        raw_output = self.inference(formatted_prompt, **pipe_twargs)
        
        # try to extract the generated text (DEPRECATED - as extraction happens in inference method)
        formatted_response = raw_output

        # prettify the output, clean artifacts, remove sentences etc.
        response = self.__post_process__(formatted_prompt, formatted_response, cut_unfinished)

        # Append newly received input, output tuple to context
        self.context[sessionId].append((input_text, response))

        return response

    def generate (self, input_text:str, max_new_tokens:int=128, echo: bool=True) -> str:

        '''
        Alternative forward propagation.
        Generates output directly from model.

        [Parameters]
        input_text          User input as str.
        max_new_tokens      The max. tokens to generate.
        echo                Whether user input should be embedded in output.

        [Return] 
        The model output as string. 
        '''

        inputs = self.tokenizer(input_text, return_tensors="pt")

        # pipe inputs to correct device
        inputs = inputs.to(self.device)

        # generate some outputs 
        outputs = self.model.generate(**inputs, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=max_new_tokens)

        # decode tensor object -> output string 
        outputString = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # remove the prepended input
        if not echo:
            outputString = outputString.replace(input_text, '')

        return outputString
    
    def inference (self, input_text: str, **pipe_twargs) -> str:

        '''
        Inference of input through model using the selected pipeline.
        '''
        # print('PROMPT:', input_text)
        if self.llm_base_module == 'vllm':
            if 'stop' not in pipe_twargs:
                pipe_twargs['stop'] = ['<|eot_id|>', '</s>']
            return self.pipe([input_text], SamplingParams(**pipe_twargs))[0].outputs[0].text        
        elif self.llm_base_module == 'llama.cpp':
            return self.pipe(input_text, **pipe_twargs)['choices'][0]['text']
        else:
            return self.pipe(input_text, **pipe_twargs)[0]['generated_text']

    def batch_inference (self, *input_text: str, **pipe_twargs) -> list[str]:

        '''
        Batch inference with vLLM.
        All functionalities are identical to BaseClient.inference.
        '''
        
        if self.llm_base_module == 'vllm':
            if 'stop' not in pipe_twargs:
                pipe_twargs['stop'] = ['<|eot_id|>', '</s>']
            outputs = self.pipe(list(input_text), SamplingParams(**pipe_twargs))      
            collected_responses = []
            for output in outputs:
                collected_responses.append(output.outputs[0].text)
            return collected_responses
        raise ValueError('Batch inference is only possible if vLLM is installed.')

    def loadModel (self, model_file:str|None=None, hugging_face_path:str|None=None, device:str='gpu', device_id:int=0, **twargs) -> bool:

        '''
        Loads model onto selected device, up to a specific GPU.

        device          device gpu/cuda, cpu
        device_id       0 (default), 1, ...
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

            # GGUF quantized files
            try:
                if 'gguf' in hugging_face_path.lower():

                    self.log(f'GGUF format detected, try to onboard {hugging_face_path} with llama.cpp ...', label='⚙️')
                    
                    # llama.cpp model equivalent
                    self.model = Llama.from_pretrained(
                        repo_id=hugging_face_path,
                        filename=model_file,
                        chat_format=self.chat_format,
                        main_gpu=device_id,
                        n_gpu_layers=-1,
                        verbose=False, # keep enabled otherwise it might cause an exception: https://github.com/abetlen/llama-cpp-python/issues/729
                        **twargs
                    )

                    return True
            except:
                print_exc()

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

            # if gpu approach fails set the torch default device to CPU
            # before trying to attempt loading the model on CPU
            self.selectDevice('cpu')


        # ==== CPU approach ====
        
        # Load with llama.cpp first (the best performance approach)
        try:

            self.log(f'try loading {hugging_face_path} with llama.cpp ...', label='⚙️')

            # [ctransformers deprecated]
            # self.model = ctransformers.AutoModelForCausalLM.from_pretrained(hugging_face_path, model_file=model_file, hf=True, **twargs)
            # self.tokenizer = ctransformers.AutoTokenizer.from_pretrained(self.model)

            # ---- llama.cpp ----
            # get the model
            self.model = Llama.from_pretrained(
                n_gpu_layers=-1,
                repo_id=hugging_face_path,
                filename=model_file,
                chat_format=self.chat_format,
                # tokenizer=LlamaHFTokenizer.from_pretrained(hugging_face_path),
                verbose=True # keep enabled otherwise it might cause an exception: https://github.com/abetlen/llama-cpp-python/issues/729
            )

            # isolate the tokenizer function
            self.tokenizer = self.model.tokenize

            # for llama.cpp the model can be used as pipe
            # see: https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#high-level-api
            # (this block is being done in __init__ already)
            # self.pipe = self.model

            # re-write the base module
            self.llm_base_module = 'llama.cpp'

            self.log(f'Successfully loaded {hugging_face_path} with llama.cpp on CPU!', label='✅')
            return True
        
        except:

            print_exc()
        
        # finally try to load with classic transformers
        try:

            self.log('fallback loading with transformers on CPU using provided arguments ...', label='⚙️')

            # default transformers
            self.model = transformers.AutoModelForCausalLM.from_pretrained(hugging_face_path, device_map="cpu", **twargs)

            # extract tokenizer
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(hugging_face_path, use_fast=True)
            self.log(f'Successfully loaded {hugging_face_path} on CPU', label='✅')
            return True

        except:

            print_exc()
        
        self.log(f'failed loading {hugging_face_path} onto CPU.', label='🛑')
        
        return False

    def log (self, *stdout: any, label='info', color='c') -> None:

        '''
        Logs args to console.
        '''

        if self.silent:
            return

        header = f"{__colors__[color]}{label.upper()}{__colors__['nc']} "
        print(header, *stdout)

    def reset (self) -> None:

        '''
        Resets the the parameters of the model, clears the current device cache and garbage.
        '''

        # reset weights to default
        try:
            # transformers
            self.model.reset_parameters()
        except:
            try:
                # llama.cpp
                self.model.reset()
            except:
                self.log('cannot reset model:', format_exc(), label='🛑')

        # clear the cache from device
        torch.cuda.empty_cache()
        gc.collect()
    
    def newConversation (self, id: str, username: str, char_tags: list[str]=['helpful'], scenario: None|str=None) -> None:

        '''
        Initializes a new context for conversation in client.context.
        The new conversation will be auto-labeled with an id.
        Note: this method does not verify if the id exists already and will simply override.
        '''
        
        # First set the beginning system prompt, from 
        # provided scenario or construct scenrio from char_tags
        if scenario:
            sys_prompt = scenario
        else:
            if self.chat_format == 'llama-2':
                sys_prompt = f"""This is a dialog, where a user interacts with {self.name}.\n {self.name} is {', '.join(char_tags)}, and is aware of his limits. \n{username}: Hello, who are you?\n{self.name}: Hello! I am **{self.name}** How can I assist you today?"""
            elif self.chat_format == 'llama-3':
                sys_prompt = f"""Your name is {self.name}, you are an assistant which is {', '.join(char_tags)}."""

        # Initialize new context with sessionId
        self.log(f'Start new conversation {id}', label='🗯️')
        self.context[id] = [self.__format_prompt__(sys_prompt, system_prompt=True)]

    def newSessionId (self) -> str:

        '''
        Generates a new random and unique session id from ns timestamp.

        [Return]
        Session id as string.
        '''
        
        # generate unique chat identifier from ns timestamp
        while True:
            id = str(time_ns())
            if not id in self.context:
                return id

    def tokenize (self, string: str) -> list[int]:

        '''
        Tokenizes a string using the loaded backend tokenizer.
        [Parameters]
        string      The string to tokenize.
        [Return]
        List of tokens encoded as byte integers.
        '''

        if self.llm_base_module == 'transformers':
            return self.tokenizer.encode(string)
        elif self.llm_base_module == 'llama.cpp':
            return self.tokenizer(string.encode())
        elif self.llm_base_module == 'vllm':
            return self.tokenizer.encode(string, return_tensors="pt")

    
    # ---- Config Handling ----
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

    def updateConfig (self, **twargs) -> None:

        '''
        Should be used when setConfig was used to update the inference parameters on the fly.
        '''

        for k, v in twargs.items():

            self.config[k] = v

    # ---- Device API ----
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
            device = torch.cuda.get_device_name(self.device_id)
        
        return device
    
    def selectDevice (self, device: str, device_id: int|None=None) -> None:

        '''
        Sets the device used by torch (cuda, cpu).
        '''

        if device in ['gpu', 'cuda']:
            self.device = f'cuda:{device_id}' if device_id else 'cuda'
        else:
            self.device = 'cpu'
        
        # override torch default device
        torch.set_default_device(self.device)

    def showDevices (self) -> None:

        '''
        Independent method which scans for all devices.
        This method does not ensure that torch will be able to use them,
        and should be used for debugging to check if drivers/kernel 
        can locate and identify the hardware.
        '''
        
        try:
            device = platform.processor()
        except:
            device = 'Unknown CPU'
        self.log('-------- CPU --------', 'device')
        self.log(device, 'device')

        # gpu test if the SMIs are installed
        try:
            line_as_bytes = subprocess.check_output("rocm-smi --showproductname", shell=True)
        except:
            try:
                line_as_bytes = subprocess.check_output("nvidia-smi -L", shell=True)
            except:
                line_as_bytes = b'Unknown GPU'
        device = line_as_bytes.decode("utf-8")
        self.log('-------- GPU --------', 'device')
        self.log(device, 'device')

    # ---- Conversion Methods ----
    def __convert_twargs__ (self, twargs: dict) -> dict:

        '''
        blowtorch tries to bring simplicity through consistent naming scheme of variables.
        It converts faulty parameters from transformers language to the llm_base_module format. 
        Converts the assembled twargs element used in methods:

        - client.inference
        - client.contextInference

        to compliant kwarg for specific library.
        E.g. llama.cpp has different naming for max_new_tokens.
        '''

        if self.llm_base_module == 'llama.cpp':

            if 'max_new_tokens' in twargs:
                twargs['max_tokens'] = twargs['max_new_tokens']
                twargs.pop('max_new_tokens')
            
            if 'min_new_tokens' in twargs:
                twargs['min_tokens'] = twargs['min_new_tokens']
                twargs.pop('min_new_tokens')
            
            if 'repetition_penalty' in twargs:
                twargs['repeat_penalty'] = twargs['repetition_penalty']
                twargs.pop('repetition_penalty')
        
        if self.llm_base_module == 'vllm':

            if 'max_new_tokens' in twargs:
                twargs['max_tokens'] = twargs['max_new_tokens']
                twargs.pop('max_new_tokens')
            
            if 'min_new_tokens' in twargs:
                twargs['min_tokens'] = twargs['min_new_tokens']
                twargs.pop('min_new_tokens')

        return twargs

    def __format_prompt__ (self, input_text: str, header: str|None=None, response: str|None=None, system_prompt: bool=False) -> str:

        '''
        Sets the correct prompting format, and returns the formatted input.
        
        [Parameters]
        input_text          The user input, or system prompt if system_prompt enabled.
        system_prompt       if enabled, will return initializing system prompt.
        header              Header sets the role, e.g. 'user', 'assistant'
        response            Option to directly encode response as well.
        chat_format       Set the llama version for proper prompt encoding.
                            default: 'llama-2'
        system_prompt       Will return encoded system prompt.

        [Prompt Encodings]
        LLaMA-2:
          https://github.com/facebookresearch/llama/issues/484#issuecomment-1649286345
          https://huggingface.co/blog/llama2#how-to-prompt-llama-2
        LLaMA-3:
          https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
        '''

        if system_prompt:

            if self.chat_format == 'llama-2':

                return f'<s>[INST] <<SYS>>\n{input_text}\n<</SYS>>\n\n'        
            
            elif self.chat_format == 'llama-3':

                return f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{input_text}<|eot_id|>'
        
        if self.chat_format == 'llama-2':

            if response:

                return f'{input_text.strip()} [/INST] {response.strip()} </s><s>[INST] '

            return f'{input_text.strip()} [/INST] '
        
        
        elif self.chat_format == 'llama-3':

            if response:
                # it's important that the response header tag is always called "assistant"
                # otherwise this will cause wrong encoding and spaming output.
                # see: https://github.com/ggerganov/llama.cpp/issues/2598                                                 vvvvvvvvv
                return f'<|start_header_id|>{header}<|end_header_id|>\n\n{input_text.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response.strip()}<|eot_id|>'

            return f'<|start_header_id|>{header}<|end_header_id|>\n\n{input_text.strip()}<|eot_id|>'

    def __post_process__ (self, _input:str, _output:str, cut_unfinished:bool=True) -> str:

        '''
        Post-processing method which takes formatted _input and _output from LLM and returns a clean string output.
        This is only raw message and string processing, no transformer tags are added. 

        [Parameter]
        _input                  Initially prompted input.
        _output                 Received raw output from prompt.
        cut_unfinished          If enabled, will remove unfinished sentences at the end.
        
        [Return]
        Processed string.
        '''

        # define processed output
        processed = _output  

        # remove initial input and empty lines
        processed = processed.replace(_input, '').replace('\n\n', '') 

        # remove annoying assistant token at the beginning in llama-3
        if 'assistant\n' in processed:
            processed = processed.replace('assistant\n', '') 
        elif 'assistant' in processed[:len(self.name)+20]:
            processed = processed.replace('assistant', '', 1)

        # format-specific processing
        if self.chat_format == 'llama-2':

            # remove potential eos, bos token artifacts
            pad_tokens = ['<<SYS>>', '<</SYS>>', '[INST]', '[/INST]' '<</s>', '<<s>', '<s>', '<</s>>']
            
        elif self.chat_format == 'llama-3':
            
            # if eot token was found cut there
            if '<|eot_id|>' in processed:
                processed = processed.split('<|eot_id|>')[0]
            
            # otherwise remove all rest tokens.
            pad_tokens = ['<|eot_id|>', '<|start_header_id|>', '<|end_header_id|>']

        # remove format-specific pad tokens
        for token in pad_tokens:
            processed = processed.replace(token, '')

        # finally remove unfinished sentences
        if cut_unfinished:
            processed = self.__cutoff_unfinished_sentence__(processed)
        
        return processed
    
    def __cutoff_unfinished_sentence__ (self, text: str) -> str:

        '''
        Cuts off unfinished sentences.
        Example: "This is the first sentence. This is not," -> "This is the first sentence."

        [Parameter]
        text        Text to cut the unfinished sentence from.

        [Return]
        String with full sentences only.
        '''
        
        for i in range(len(text)-1, 0, -1):
            if text[i] in '.?!':
                return text[:i+1]
        return ''

    # ---- Benchmark Methods ----
    def bench (self, tokens: int=512) -> None:

        '''
        A quick benchmark of the loaded model.

        [Parameters]
        tokens      Number of tokens to generate.
        '''

        kwargs = self.__convert_twargs__({'max_new_tokens': tokens})

        self.log('Start benchmark ...', label='⏱️')

        # prepare a message in perfect format in case of vllm
        # as vllm will otherwise only generate a single new token
        bench_prompt = 'please write a generic long letter'
        if self.llm_base_module == 'vllm':
            bench_prompt = self.__format_prompt__(bench_prompt)

        # Perform benchmark
        stopwatch_start = perf_counter_ns()
        raw_output = self.inference(bench_prompt, **kwargs)
        stopwatch_stop = perf_counter_ns()
        

        # Get the memory usage of the current process
        memory_usage = self.ramUsage() # memory occupied by the process in total
        vram_usage = self.vramUsage() # memory allocated b torch on gpu

        # count tokens
        tokens = len(self.tokenize(raw_output))
        bytes = len(raw_output)

        # compute statistics
        duration = (stopwatch_stop - stopwatch_start) * 1e-9
        token_rate = round(tokens/duration, 3)
        tpot = round(1e3 / token_rate, 3) # time per output token in ms
        data_rate = round(bytes/duration, 3)
        bit_rate = round(data_rate*8, 3)
        duration = round(duration, 3)

        # print results
        print('\n-------- benchmark results --------')
        print(
            f'Model: {self.hugging_face_path}',
            f'\nDevice ({self.device}): {self.getDeviceName()}',
            f'\nDevice ID used: {0 if self.device == "cpu" else torch.cuda.current_device()}',
            f'\nRAM Usage: {memory_usage[0]} {memory_usage[1]}',
            f'\nvRAM Usage: {vram_usage[0]} {vram_usage[1]}',
            f'\nMax. Token Window: {tokens}',
            f'\nTokens Generated: {tokens}',
            f'\nBytes Generated: {bytes } bytes'
            f'\nOutput Token Rate: {token_rate} tokens/s', 
            f'\nData Rate: {data_rate} bytes/s',
            f'\nBit Rate: {bit_rate} bit/s',
            f'\nTPOT: {tpot} ms/token',
            f'\nTotal Gen. Time: {duration} s'
        )
    
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

class client ( BaseClient ):

    '''
    Alias for BaseClient.
    '''

    def __init__(self, model_file = None, hugging_face_path = None, attention = False, chat_format = 'llama-3', device = 'gpu', device_id = 0, name = None, verbose = True, silent = False, **twargs):
        super().__init__(model_file, hugging_face_path, attention, chat_format, device, device_id, name, verbose, silent, **twargs)




# ---- exposers ----
class console:

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

            # update token size
            maxNewTokens = data['maxNewTokens']
            self.__client__.updateConfig(max_new_tokens=maxNewTokens)
            
            # do inference
            message = data['message']
            output = self.__client__.contextInference(
                message, 
                sessionId=sessionId
            )

            self.__client__.log('send:', output, label='📨')

            # add output to data
            response['data']['message'] = output

        # prepare response
        encoded = json.dumps(response).encode('utf-8')

        # send
        self.wfile.write(encoded)

class webUI:

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