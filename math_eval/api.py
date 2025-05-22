from openai import OpenAI, AsyncOpenAI
import asyncio


def construct_prompt(problem, answer, generation):
    prompt = f"""You are a mathematical answer validator. You will be provided with a mathematical problem and you need to compare the answer in the reference solution, and the final answer in a model's solution to determine if they are equivalent, even if formatted differently.

PROBLEM:

{problem}

REFERENCE SOLUTION:

{answer}

MODEL'S SOLUTION:

{generation}

Focus ONLY on comparing the final mathematical answer provided by the model while ignoring differences in:

- Formatting (e.g., \\boxed{{}} vs plain text)
- Multiple choice formatting (e.g., "A" vs full solution)
- Order of coordinate pairs or solutions
- Equivalent mathematical expressions or notation variations
- If the model's answer is nonsense, return "Verdict: AMBIGUOUS"

Start with a brief explanation of your comparison (2-3 sentences). Then output your final answer in one of the following formats:

- "Verdict: EQUIVALENT"
- "Verdict: DIFFERENT"
- "Verdict: AMBIGUOUS"
"""
    return prompt


async def async_query_openai(query, request_id, semaphore):
    async with semaphore:
        retries = 0
        max_retries = 10
        while retries < max_retries:
            try:
                # replace with your own api key
                aclient = AsyncOpenAI(
                    base_url=" ",
                    api_key=" ",
                )
                completion = await aclient.chat.completions.create(
                    model="deepseekr1",
                    messages=[{"role": "user", "content": query}],
                    temperature=0,
                    max_tokens=2048
                )
                return request_id, completion.choices[0].message.content

            except Exception as e:
                retries += 1
                print(f"request ID {request_id} error: {e}, retry attempt {retries}")
                if retries >= max_retries:
                    print(f"request ID {request_id} exceeded maximum retry attempts, returning error")
                    return request_id, "request error"
                await asyncio.sleep(10)


async def async_process_queries(queries, concurrency_limit):
    semaphore = asyncio.Semaphore(concurrency_limit)
    tasks = [asyncio.create_task(async_query_openai(query, i, semaphore))
             for i, query in enumerate(queries)]
    total = len(tasks)
    completed = 0
    results = [None] * total

    for future in asyncio.as_completed(tasks):
        request_id, result = await future
        results[request_id] = result
        completed += 1
        print(f"Progress: {completed}/{total} queries processed, current completed task ID: {request_id}")
    return results