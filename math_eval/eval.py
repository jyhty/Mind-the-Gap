import os
import json
import argparse
from tqdm import tqdm
from util import *
from vllm import LLM, SamplingParams
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="amc23", help="Comma-separated dataset names")
parser.add_argument("--n_sampling", type=int, default=4, help="Number of sampling")
parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens per generation")
parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature")
parser.add_argument("--model_name", type=str, help="Path to the model")
parser.add_argument("--output_dir", type=str, help="Output directory")
parser.add_argument("--few_shot", action="store_true", help="Enable few-shot mode", default=False)
args = parser.parse_args()

available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")

data_names = args.data_name.split(",")

llm = LLM(
    model=args.model_name,
    tensor_parallel_size=len(available_gpus),
    trust_remote_code=False,
    gpu_memory_utilization=0.95,
    max_model_len=4096,
    dtype="bfloat16"
)

for data_name in data_names:
    examples = load_data(data_name, 'test')
    samples = []
    for idx, example in tqdm(enumerate(examples), total=len(examples)):
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
        }
        samples.append(sample)
    input_prompts = [sample["question"] for sample in samples for _ in range(args.n_sampling)]
    if not args.few_shot:
        input_prompts = [
            f"""System: You are a math problem solver. You should think step by step.\nHuman: {prompt}\nAssistant:""" for prompt in input_prompts

            # Qwen Style
            # "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
            # + prompt
            # + "<|im_end|>\n<|im_start|>assistant\n" for prompt in input_prompts

            # R1 Style
            # "Please reason step by step, and put your final answer within \\boxed{}.\nUser: "
            # + prompt
            # + "\nAssistant: <think>" for prompt in input_prompts
        ]
    else:
        input_prompts = [f"""Q: {EXAMPLES[0][0]}
A: Let's think step by step. {EXAMPLES[0][1]}

Q: {EXAMPLES[1][0]}
A: Let's think step by step. {EXAMPLES[1][1]}

Q: {EXAMPLES[2][0]}
A: Let's think step by step. {EXAMPLES[2][1]}

Q: {EXAMPLES[3][0]}
A: Let's think step by step. {EXAMPLES[3][1]}

Q: {prompt}
A: Let's think step by step.
""" for prompt in input_prompts]
    outputs = llm.generate(
        input_prompts,
        SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            skip_special_tokens=False,
        ),
    )

    outputs = [output.outputs[0].text for output in outputs]
    for i, sample in enumerate(samples):
        output = outputs[i * args.n_sampling : (i + 1) * args.n_sampling]
        sample.update({"outputs": output})

    extraction_target = (ExprExtractionConfig(), LatexExtractionConfig())
    results = []
    for sample in samples:
        llm_outputs = sample["outputs"]
        correct_answer = sample["gt"]
        temp = []
        for llm_output in llm_outputs:
            gold = parse(f"${correct_answer}$", extraction_config=extraction_target)
            answer = parse(llm_output, extraction_config=extraction_target)
            result = verify(gold, answer)
            temp.append(result)
            sample.update({"eval": temp})
            results.append(result)

    accuracy = sum(results) / len(results)
    print(f"{data_name} initial accuracy using Math-Verify", accuracy)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/eval.json"
    acc_file = f"{output_dir}/{data_name}/acc.txt"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(samples, f, indent=4)
    with open(acc_file, "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
