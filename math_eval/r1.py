import time
import pandas as pd
import os
import pickle
import argparse
import asyncio
from api import *
parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, required=True)
args = parser.parse_args()
file_name = args.file_name

df = pd.read_json(file_name)

texts, correctness = [], []
for i in range(len(df)):
    row = df.iloc[i]
    question = row["question"]
    llm_outputs = row["outputs"]
    correct_answer = row["gt"]
    evals = row["eval"]
    for j, llm_output in enumerate(llm_outputs):
        if not evals[j]:
            text = construct_prompt(question, correct_answer, llm_output)
            texts.append(text)
            correctness.append(False)
        else:
            correctness.append(True)
print(len(texts))

CONCURRENCY_LIMIT = 32

async def main():
    queries = texts
    start_time = time.time()
    results = await async_process_queries(queries, CONCURRENCY_LIMIT)
    end_time = time.time()
    print(f"Evaluating Time: {end_time - start_time:.2f} seconds")
    return results

output_dir = os.path.dirname(file_name)
results = asyncio.run(main())
final_correctness = []

temp = 0
for i in range(len(df)):
    row = df.iloc[i]
    evals = row["eval"]
    for j in range(len(evals)):
        if evals[j]:
            final_correctness.append(True)
            continue
        if "Verdict: EQUIVALENT" in results[temp]:
            final_correctness.append(True)
            df.at[i, "eval"][j] = True
        else:
            final_correctness.append(False)
        temp += 1

final_accuracy = sum(final_correctness) / len(final_correctness)
print(final_accuracy)

def save_to_dir(filename, data):
    path = os.path.join(output_dir, filename)
    with open(path, 'wb') as f:
        pickle.dump(data, f)

save_to_dir('double_check.pkl', results)

accuracy_path = os.path.join(output_dir, 'final_acc.txt')
with open(accuracy_path, 'w') as f:
    f.write(f"{final_accuracy:.4f}")
print(f"Final Accuracy: {final_accuracy:.4f}")

eval_updated_path = os.path.join(output_dir, 'eval_updated.json')
df.to_json(eval_updated_path, orient='records', indent=4)
print(f"Updated eval.jsonl saved to: {eval_updated_path}")
