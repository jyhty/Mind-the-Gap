import os
import json
import pandas as pd


def load_data(data_name, split, data_dir="./data"):
    data_file = f"{data_dir}/{data_name}/{split}.json"
    if os.path.exists(data_file):
        df = pd.read_json(data_file)
        return df


def construct_prompt(data_name):
    df = load_data(data_name, "test")
    if data_name in [
        "folio",
        "pw",
        "ruletaker",
    ]:
        prompts = []
        for i in range(len(df)):
            question = f"""Given:\n{df.iloc[i]["context"]}
Conclusion:\n{df.iloc[i]["conclusion"]}
Is the conclusion correct based on the known conditions?
A: False
B: True
which option is correct?"""
            st = "You are a math problem solver. You should think step by step."
            prompt = f"""System: {st}\nHuman: {question}\nAssistant:"""
            prompts.append(prompt)
        return prompts
    elif data_name in [
        "logicqa",
        "reclor",
    ]:
        if data_name == "logicqa":
            prompts = []
            for i in range(len(df)):
                st = "You are a math problem solver. You should think step by step."
                question = df.iloc[i]["context"] + "\n" + df.iloc[i]["question"]+ "\n" + df.iloc[i]["options"]+ "\n" +"which option is correct?"
                prompt = f"""System: {st}\nHuman: {question}\nAssistant:"""
                prompts.append(prompt)
            return prompts
        if data_name == "reclor":
            prompts = []
            for i in range(len(df)):
                st = "You are a math problem solver. You should think step by step."
                options = df.iloc[i]["answers"]
                temp = f"""A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}"""
                question = df.iloc[i]["context"] + "\n" + df.iloc[i]["question"]+ "\n" + temp+ "\n" +"which option is correct?"
                prompt = f"""System: {st}\nHuman: {question}\nAssistant:"""
                prompts.append(prompt)
            return prompts