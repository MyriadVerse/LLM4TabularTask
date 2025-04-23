# process_query.py
import base64
import math
import pickle
import sys
import json
import pandas as pd
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from supervised_fine_tune import PROMPT_DICT, types
from inference_rel_extraction_col_type import main


def generate_prompt_to_llm(instruction, question, input_seg=None):
  if input:
    return PROMPT_DICT["prompt_input"].format(instruction=instruction, input_seg=input_seg, question=question)
  else:
    return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
  

def build_generator(
    item, model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=64, use_cache=True, max_token_count=6144
):
    def response(item):
        prompt = generate_prompt_to_llm(instruction = item["instruction"], input_seg = item["input_seg"], question = item["question"])
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


def generate_prompt(task, table, query):
    """generate task-specific prompt"""
    if task == "Column Type Annotation":
        # serilize table headers
        col_head = "[TAB] col: | "
        for col_name in table.columns:
            col_head += f"{col_name} | "

        # serilize table data
        table_cell = ""
        min_row_len = table.count().min() 
        min_row_len = min(min_row_len, 10)
        for row_idx in range(min_row_len):
            row_data = table.iloc[row_idx].apply(str) 
            example_values = " | ".join(row_data)
            
            if row_idx == 0:
                table_cell += f"row {row_idx+1}: | {example_values} |"
            else:
                table_cell += f" [SEP] row {row_idx+1}: | {example_values} |"

        # highlight target column
        # column_index = int(query[-1])
        column_index = table.columns.get_loc(query.split()[-1]) 
        entities = ", ".join([f"<{item.strip()}>" for item in table.iloc[:, column_index].dropna().apply(str)][:min_row_len])

        table_prompt = {}
        table_prompt["instruction"] = "This is a column type annotation task. The goal for this task is to choose the correct semantic type for one selected column of the table from the given candidates."
        table_prompt["input_seg"] = col_head + table_cell
        table_prompt["question"] = f"The column '{table.columns[column_index]}' contains following entities: {entities}, etc. The column type candidates are: {types}. What are the correct semantic column type for this column (column name: {table.columns[column_index]}; entities: {entities}, etc)?"
    
    elif task == "Hybrid Question Answering":
        # serilize table headers
        col_head = "[TAB] col: | "
        for col_name in table.columns:
            col_head += f"{col_name} | "

        # serilize table data
        table_cell = ""
        min_row_len = table.count().min() 
        min_row_len = min(min_row_len, 10)
        for row_idx in range(min_row_len):
            row_data = table.iloc[row_idx].apply(str) 
            example_values = " | ".join(row_data)
            table_cell += f" [SEP] | {example_values} |"

        table_prompt = {}
        table_prompt["instruction"] = "This is a hybrid question answering task. The goal of this task is to answer the question given tables."
        table_prompt["input_seg"] = col_head + table_cell
        table_prompt["question"] = f"You can refer to the table when you answer the question. The question: {query}?"
        
    return table_prompt

def process_query(task, table_str, query):
    """处理查询的主函数"""
    try:
        # load table
        table_bytes = base64.b64decode(table_str.encode('utf-8'))
        df = pickle.loads(table_bytes)

        # generate prompt
        prompt = generate_prompt(task, df, query)

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
            respond = build_generator(prompt, model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=128, use_cache=True, max_token_count=8192 if 8192 > orig_ctx_len else orig_ctx_len)   
            output = respond(prompt)
            return output

    except Exception as e:
        return f"Error processing query: {str(e)}"

if __name__ == "__main__":
    task = sys.argv[1]
    table_str = sys.argv[2]
    query = sys.argv[3]
    
    result = process_query(task, table_str, query)
    print(result)