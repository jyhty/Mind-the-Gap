import os
import json
import argparse
from tqdm import tqdm
from util import *
from vllm import LLM, SamplingParams
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="logicqa", help="Comma-separated dataset names")
parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens per generation")
parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature")
parser.add_argument("--model_name", type=str, help="Path to the model")
parser.add_argument("--output_dir", type=str, help="Output directory")
args = parser.parse_args()

available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")

data_names = args.data_name.split(",")

llm = LLM(
    model=args.model_name,
    tensor_parallel_size=len(available_gpus),
    trust_remote_code=False,
    gpu_memory_utilization=0.95,
    max_model_len=4096
)

for data_name in data_names:
    input_prompts = construct_prompt(data_name)
    outputs = llm.generate(
        input_prompts,
        SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            skip_special_tokens=False,
        ),
    )

    results = [output.outputs[0].text for output in outputs]
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
    with open(f'{output_dir}/{data_name}/results.pkl', 'wb') as f:
        pickle.dump(results, f)

