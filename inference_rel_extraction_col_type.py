import ast
import os
import json
import sys
import math
import time
import torch, gc
import argparse
# import textwrap
import transformers
# from peft import PeftModel
from transformers import GenerationConfig
# from llama_attn_replace import replace_llama_attn
from supervised_fine_tune import PROMPT_DICT
from tqdm import tqdm
# from queue import Queue
# from threading import Thread
# import gradio as gr

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--input_data', type=str, default=None, help='input data')
    args = parser.parse_args()
    return args

def generate_prompt(instruction, question, input_seg=None):
  if input:
    return PROMPT_DICT["prompt_input"].format(instruction=instruction, input_seg=input_seg, question=question)
  else:
    return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)


def build_generator(
    item, model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=64, use_cache=True, max_token_count=6144
):
    def response(item):
        prompt = generate_prompt(instruction = item["instruction"], input_seg = item["input_seg"], question = item["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        token_count = inputs['input_ids'].size(1)
        
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
            pad_token_id=tokenizer.eos_token_id
        )
        out = tokenizer.decode(output[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)

        out = out.split(prompt)[1].strip()
        return out

    return response

def main(args):
    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        "osunlp/TableLlama",
        cache_dir="./cache",
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and 8192 > orig_ctx_len:
        scaling_factor = float(math.ceil(8192 / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    print("Loading model...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "osunlp/TableLlama",
        config=config,
        cache_dir="./cache",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.resize_token_embeddings(32001)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "osunlp/TableLlama",
        cache_dir="./cache",
        model_max_length=8192 if 8192 > orig_ctx_len else orig_ctx_len,
        padding_side="left",
        use_fast=False,
    )

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    with torch.no_grad():  
        print("Loading data...")
        item = json.loads(args.input_data)
        print('llm')
        respond = build_generator(item, model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=128, use_cache=True, max_token_count=8192 if 8192 > orig_ctx_len else orig_ctx_len)   
        output = respond(item)
        print(output)
        
if __name__ == "__main__":
    print("Loading config...")
    args = parse_config()
    main(args)
    

