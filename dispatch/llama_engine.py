#!/usr/bin/env python3
from typing import Optional
import os

class LlamaEngine:
    def __init__(self, n_threads=4, n_predict=256):
        self._model = None
        self._n_threads = n_threads
        self._n_predict = n_predict

    def load(self, model_path):
        if not os.path.isfile(model_path):
            print('[Llama] Model not found:', model_path)
            return False
        try:
            from llama_cpp import Llama
            self._model = Llama(
                model_path=model_path, n_ctx=2048,
                n_threads=self._n_threads, verbose=False)
            print('[Llama] Model ready:', model_path)
            return True
        except Exception as e:
            print('[Llama] Load failed:', e)
            return False

    def chat(self, prompt, system_prompt=''):
        if self._model is None:
            raise RuntimeError('[Llama] No model loaded')
        im_s, im_e = '<|im_start|>', '<|im_end|>'
        nl = chr(10)
        parts = []
        if system_prompt:
            parts.append(im_s + 'system' + nl + system_prompt + im_e)
        parts.append(im_s + 'user' + nl + prompt + im_e)
        parts.append(im_s + 'assistant' + nl)
        full = nl.join(parts)
        try:
            result = self._model.create_completion(
                full, max_tokens=self._n_predict,
                temperature=0.7, stop=[im_e, im_s])
            return result['choices'][0]['text'].strip()
        except Exception as e:
            return '[Llama error: ' + str(e) + ']'

    def unload(self):
        self._model = None

    def is_loaded(self):
        return self._model is not None
