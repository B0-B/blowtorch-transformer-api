#!/usr/bin/env python3

'''
This module was assembled from various documentations and posts, it can load 
huggingface paths of text generation models and pipeline them for inference. 
If not furter specified the default huggingface model will 
be palmyra-small (128M): https://huggingface.co/Writer/palmyra-small.
'''

import torch
import transformers, ctransformers
from traceback import print_exc

class flashModel:

    '''
    model_file          e.g. llama-2-7b-chat.Q2_K.gguf
                        specific file from card
    huggingFacePath     e.g. Writer/palmyra-small
    gpu_layers          number of layers to offload gpus (0 for pure cpu usage)
    '''

    def __init__ (self, model_file:str|None=None, hugging_face_path:str|None=None, device:str='gpu', gpu_layers:int=0, **kwargs) -> None:

        # load pre-trained model and input tokenizer
        # default will be palmyra-small 128M parameters
        if not hugging_face_path:
            hugging_face_path = "Writer/palmyra-small"
        
        # extract the AI name
        self.name = hugging_face_path.split('/')[-1]

        # collect all arguments for model
        _kwargs = {}
        if model_file:
            _kwargs['model_file'] = model_file
        if gpu_layers:
            _kwargs['gpu_layers'] = gpu_layers
        if 'load_in_8bit' in kwargs and kwargs['load_in_8bit']:
            _kwargs['load_in_8bit'] = True
        for k,v in kwargs.items():
            _kwargs[k] = v
        
        # -- load model and tokenizer --
        try:

            # some models can be quantized to 8-bit precision
            print('info: try loading transformers with assembled arguments ...')

            # default transformers
            self.model = transformers.AutoModelForCausalLM.from_pretrained(**_kwargs)
            # extract tokenizer
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(hugging_face_path)

        except:

            # ctransfomers for builds with GGUF weight format
            try:

                print('info: try loading with ctransformers ...')
                
                self.model = ctransformers.AutoModelForCausalLM.from_pretrained(hugging_face_path, model_file=model_file, gpu_layers=gpu_layers, hf=True, **kwargs)
                self.tokenizer = ctransformers.AutoTokenizer.from_pretrained(self.model)

            except:

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

        # create pipeline
        self.pipe = transformers.pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

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
    
    def chat (self, username: str='human', **pipe_kwargs) -> None:

        '''
        A text-to-text cli loop.
        Helpful to interfere with the model via command line.
        '''

        while True:

            try:

                inputs = input(f'{username}: ')
                
                # get raw string output
                raw_output = self.inference(inputs, **pipe_kwargs)

                # post-processing
                processed = raw_output[0]['generated_text'].replace(inputs, '').replace('\n\n', '')

                print(f'\n{self.name}: {processed}')
                # print('\n' + self.name + ':', self.inference(inputs, **pipe_kwargs))

            except KeyboardInterrupt:
                
                break

            except Exception as e:

                print_exc()

    def inference (self, input_text:str, max_new_tokens:int=64, **pipe_kwargs) -> str:

        '''
        Inference of input through model using the transformer pipeline.
        '''

        return self.pipe(input_text, max_new_tokens=max_new_tokens, **pipe_kwargs)

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

    def setDevice (self, device: str) -> None:

        '''
        Sets the device torch should use (cuda, cpu).
        '''

        if device in ['gpu', 'cuda']:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

if __name__ == '__main__':

    flashModel('llama-2-7b-chat.Q2_K.gguf', 'TheBloke/Llama-2-7B-Chat-GGUF', 'cpu', model_type="llama").chat(max_new_tokens=512, do_sample=False, temperature=0.8, repetition_penalty=1.1)
    # flashModel(hugging_face_path='TheBloke/Llama-2-7B-Chat-GGML', device='cpu', model_type="llama").cli(max_new_tokens=64, do_sample=True, temperature=0.8, repetition_penalty=1.1)
    # flashModel(device='cpu').cli() # run small palmyra model for testing