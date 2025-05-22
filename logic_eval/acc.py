import os
import json
import argparse
from tqdm import tqdm
from util import *
from vllm import LLM, SamplingParams
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="logicqa", help="Comma-separated dataset names")
parser.add_argument("--output_dir", type=str, help="Output directory")
args = parser.parse_args()

available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")

data_names = args.data_name.split(",")

llm = LLM(
    model="xFinder-qwen1505",
    tensor_parallel_size=len(available_gpus),
    trust_remote_code=False,
    gpu_memory_utilization=0.95,
    max_model_len=8000
)

PROMPT_TEMPLATE = {
    "xFinder-qwen1505":
        """<|System|>:{system}
<|User|>:{input}
<|Bot|>:""",
    "xFinder-llama38it":
        """<|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
}

def compare(letter, number):
    return mapping.get(letter) == number

for data_name in data_names:
    df = pd.read_json(f"./data/{data_name}/test.json")
    output_dir = args.output_dir
    with open(f'./outputs/{output_dir}/{data_name}/results.pkl', 'rb') as f:
        loaded_results = pickle.load(f)
    texts = []
    if data_name in [
        "folio",
        "pw",
        "ruletaker",
    ]:
        for i in range(len(loaded_results)):
            question = f"""Given:\n{df.iloc[i]["context"]}
Conclusion:\n{df.iloc[i]["conclusion"]}
Is the conclusion correct based on the known conditions?
A: False
B: True
which option is correct?"""
            llm_output = loaded_results[i]
            standard_answer_range = "[['A', 'False'], ['B', 'True']]"
            formatted_query = f'Question: """{question}"""\n\nOutput sentences: """{llm_output}"""\n\nAnswer range: {standard_answer_range}\n\nKey extracted answer: '
            system_prompt = "You are a help assistant tasked with extracting the precise key answer from given output sentences."
            text = PROMPT_TEMPLATE["xFinder-qwen1505"].format(system=system_prompt, input=formatted_query)
            texts.append(text)
    elif data_name in [
        "logicqa",
        "reclor",
    ]:
        if data_name == "logicqa":
            for i in range(len(loaded_results)):
                options = df.iloc[i]["options"].split("\n")
                question = df.iloc[i]["context"] + "\n" + df.iloc[i]["question"]+ "\n" + df.iloc[i]["options"]+ "\n" +"which option is correct?"
                llm_output = loaded_results[i]
                standard_answer_range = f"[['A', '{options[0]}'], ['B', '{options[1]}'], ['C', '{options[2]}'], ['D', '{options[3]}']]"
                formatted_query = f'Question: """{question}"""\n\nOutput sentences: """{llm_output}"""\n\nAnswer range: {standard_answer_range}\n\nKey extracted answer: '
                system_prompt = "You are a help assistant tasked with extracting the precise key answer from given output sentences."
                text = PROMPT_TEMPLATE["xFinder-qwen1505"].format(system=system_prompt, input=formatted_query)
                texts.append(text)
        if data_name == "reclor":
            for i in range(len(loaded_results)):
                options = df.iloc[i]["answers"]
                temp = f"""A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}"""
                question = df.iloc[i]["context"] + "\n" + df.iloc[i]["question"]+ "\n" + temp+ "\n" +"which option is correct?"
                llm_output = loaded_results[i]
                standard_answer_range = f"[['A', '{options[0]}'], ['B', '{options[1]}'], ['C', '{options[2]}'], ['D', '{options[3]}']]"
                formatted_query = f'Question: """{question}"""\n\nOutput sentences: """{llm_output}"""\n\nAnswer range: {standard_answer_range}\n\nKey extracted answer: '
                system_prompt = "You are a help assistant tasked with extracting the precise key answer from given output sentences."
                text = PROMPT_TEMPLATE["xFinder-qwen1505"].format(system=system_prompt, input=formatted_query)
                texts.append(text)

            
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=8000,
    )
    outputs = llm.generate(texts, sampling_params)
    extracted_answer = [o.outputs[0].text for o in outputs]

    mapping = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
    }

    cnt = 0
    if data_name == "reclor":
        for i in range(len(extracted_answer)):
            if compare(extracted_answer[i], int(df.iloc[i]["label"])):
                cnt+=1
    else:
        for i in range(len(extracted_answer)):
            if compare(extracted_answer[i], int(df.iloc[i]["answer"])):
                cnt+=1
    abnormal = 0
    for i in range(len(extracted_answer)):
        if (len(extracted_answer[i])) != 1:
            abnormal += 1

    print("correct:", cnt)  # True
    print("invalid:", abnormal)
    print(cnt/len(extracted_answer))

    with open(f'./outputs/{output_dir}/{data_name}/acc.txt', 'w') as f:
        f.write(f"correct: {cnt}\n")
        f.write(f"total: {extracted_answer}\n")
        f.write(f"Accuracy: {cnt/len(extracted_answer)}\n")
        f.write(f"invalid: {abnormal}\n")

