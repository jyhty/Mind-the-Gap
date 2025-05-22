from vllm import LLM
import os
import pickle
import re
from tqdm import tqdm
import pandas as pd
from collections import Counter
import random
import json
from transformers import AutoTokenizer
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct")
from datasets import load_dataset
save_path = "NuminaMath-CoT"
dataset = load_dataset(save_path)
filtered_train = dataset["train"]
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
llm = LLM(
    model="Qwen/Qwen2.5-Math-PRM-7B/",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.95
)

file_path = 'bridge_results.pkl' # bridge results

with open(file_path, 'rb') as file:
    loaded_results = pickle.load(file)


def extract_missing_steps(text):
    pattern = re.compile(
        r'Missing Step (\d+)：\s*'
        r'The missing step should be placed between Step (\d+) and Step (\d+)\.\s*'
        r'The missing step is:\s*(.*?)'
        r'(?=\s*Missing Step \d+：|\Z)',
        re.DOTALL
    )
    matches = pattern.findall(text)
    results = []
    for match in matches:
        a, x, y, z = match
        results.append({
            'a': a.strip(),
            'x': x.strip(),
            'y': y.strip(),
            'z': z.strip()
        })
    return results


def filter_results(results):
    filtered_results = []
    for result in results:
        x = int(result['x'])
        y = int(result['y'])
        z = result['z']
        if x + 1 == y and not z.startswith('####'):
            filtered_results.append(result)
    return filtered_results


def sort_results_by_x(results):
    sorted_results = sorted(results, key=lambda result: int(result['x']))
    return sorted_results


def process_text(text):
    results = extract_missing_steps(text)
    filtered_results = filter_results(results)
    sorted_results = sort_results_by_x(filtered_results)
    return sorted_results


def process(i):
    query = filtered_train[i]['problem']
    response = filtered_train[i]['solution']
    data = response.split("\n\n")
    result = "\n".join([f"step{i + 1}:\n{item}" for i, item in enumerate(data)])
    temp = process_text(loaded_results[i])
    insert_pos = []
    for i in range(len(temp)):
        x = temp[i]['x']
        missing_step = temp[i]['z']
        data.insert(int(x) + i, missing_step)
        insert_pos.append(int(x) + i)
    insert_result = "\n".join([f"step{i + 1}:\n{item}" for i, item in enumerate(data)])
    output = "\n".join([item for item in data])
    sy = "You are a math problem solver. You should think step by step."
    return query, data, insert_pos


query, m, insert_pos = [], [], []
for i in tqdm(range(len(filtered_train))):
    t1, t2, t3 = process(i)
    query.append(t1)
    m.append(t2)
    insert_pos.append(t3)

texts = []
for i in tqdm(range(len(filtered_train))):
    data = {
        "system": "You are a math problem solver. You should think step by step.",
        "query": query[i],
        "response": m[i]
    }

    messages = [
        {"role": "system", "content": data['system']},
        {"role": "user", "content": data['query']},
        {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    texts.append(text)

results = []
for i in tqdm(range(len(filtered_train))):
    try:
        outputs = llm.encode(texts[i], use_tqdm=False)
        results.append(outputs[0].outputs.data[:, 1].tolist())
    except:
        results.append("prm_error")
    if i%1000==0:
        with open("prm_result_temp.pkl", 'wb') as f:
            pickle.dump(results, f)

with open("prm_result.pkl", 'wb') as f:
    pickle.dump(results, f)
