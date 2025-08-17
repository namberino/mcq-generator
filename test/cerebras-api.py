# import os
# from cerebras.cloud.sdk import Cerebras
import tiktoken

# client = Cerebras(
#     # This is the default and can be omitted
#     api_key=os.environ.get("CEREBRAS_API_KEY")
# )

# stream = client.chat.completions.create(
#     messages=[
#         {
#             "role": "system",
#             "content": ""
#         }
#     ],
#     model="gpt-oss-120b",
#     stream=True,
#     max_completion_tokens=65536,
#     temperature=1,
#     top_p=1
# )
import numpy as np

INPUT_TOKEN_COUNT = np.array([], dtype=int)
OUTPUT_TOKEN_COUNT = np.array([], dtype=int)

# for chunk in stream:
# 	print(chunk.choices[0].delta.content or "", end="")
with open('../test/mcq_output.json', 'r', encoding='utf-8') as f:
	text = f.read()

def count_tokens(text: str, model_name='gpt-oss-120b', encoding_name='cl100k_base') -> int:
    """Look up model encoding; fallback to encoding_name if model not known."""
    try:
        # encoding_for_model can raise if model is unknown to tiktoken
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = None

    if enc is None:
        enc = tiktoken.get_encoding(encoding_name)

    return len(enc.encode(text))

c = count_tokens(text)
INPUT_TOKEN_COUNT = np.append(INPUT_TOKEN_COUNT, c)
print(INPUT_TOKEN_COUNT)