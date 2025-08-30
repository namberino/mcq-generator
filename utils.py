import re
import json
from typing import Dict, Any
import requests
import os
import numpy as np
import uuid
import datetime
import pathlib

#TODO: allow to choose different provider later + dynamic routing when token expired
API_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_API_KEY = os.environ['CEREBRAS_API_KEY']

HEADERS = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
JSON_OBJ_RE = re.compile(r"(\{[\s\S]*\})", re.MULTILINE)

INPUT_TOKEN_COUNT = np.array([], dtype=int)
OUTPUT_TOKEN_COUNT = np.array([], dtype=int)
TOTAL_TOKEN_COUNT = np.array([], dtype=int)
TOTAL_TOKEN_COUNT_EACH_GENERATION = np.array([])
TIME_INFOs = {}


def _post_chat(messages: list, model: str, temperature: float = 0.2, timeout: int = 60) -> str:
    payload = {"model": model, "messages": messages, "temperature": temperature}
    resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    save_to_local('test/raw_resp.json', content=data)

    #? Must update within _post_chat because it the original function for LLM generation
    update_token_count(token_usage=data['usage']) # get data['usages']['prompt_tokens'] & data['usages']['completion_tokens']
    update_time_info(time_info=data['time_info'])

    # handle various shapes
    if "choices" in data and len(data["choices"]) > 0:
        # prefer message.content
        ch = data["choices"][0]

        if isinstance(ch, dict) and "message" in ch and "content" in ch["message"]:
            return ch["message"]["content"]

        if "text" in ch:
            return ch["text"]

    print(f'Generation Time: {data["time_info"]}')
    # final fallback
    raise RuntimeError("Unexpected HF response shape: " + json.dumps(data)[:200])


def _safe_extract_json(text: str) -> dict:
    # remove triple backticks
    text = re.sub(r"```(?:json)?\n?", "", text)
    m = JSON_OBJ_RE.search(text)

    if not m:
        raise ValueError("No JSON object found in model output.")
    js = m.group(1)

    # try load, fix trailing commas
    try:
        return json.loads(js)
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*([}\]])", r"\1", js)
        return json.loads(fixed)


def generate_mcqs_from_text(
    source_text: str,
    n: int = 3,
    model: str = "gpt-oss-120b",
    temperature: float = 0.2,
) -> Dict[str, Any]:
    system_message = {
        "role": "system",
        "content": (
            "Bạn là một trợ lý hữu ích chuyên tạo câu hỏi trắc nghiệm. Luôn trả lời bằng Tiếng Việt."
            "Chỉ TRẢ VỀ duy nhất một đối tượng JSON theo đúng schema sau và không có bất kỳ văn bản nào khác:\n\n"
            "{\n"
            '  "1": { "câu hỏi": "...", "lựa chọn": {"a":"...","b":"...","c":"...","d":"..."}, "đáp án":"..."},\n'
            '  "2": { ... }\n'
            "}\n\n"
            "Lưu ý:\n"
            f"- Tạo đúng {n} mục, đánh số từ 1 tới {n}.\n"
            "- Khóa 'lựa chọn' phải có các phím a, b, c, d.\n"
            "- 'đáp án' phải là toàn văn đáp án đúng (không phải ký tự chữ cái), và giá trị này phải khớp chính xác với một trong các giá trị trong 'lựa chọn'.\n"
            "- Không kèm giải thích hay trường thêm.\n"
            "- Các phương án sai (distractors) phải hợp lý và không lặp lại."
        )
    }
    user_message = {
        "role": "user",
        "content": (
            f"Hãy tạo {n} câu hỏi trắc nghiệm từ nội dung dưới đây. Dùng nội dung này làm nguồn duy nhất để trả lời."
            "Nếu nội dung quá ít để tạo câu hỏi chính xác, hãy tạo các phương án hợp lý nhưng có thể biện minh được.\n\n"
            f"Nội dung:\n\n{source_text}"
        )
    }

    raw = _post_chat([system_message, user_message], model=model, temperature=temperature)
    parsed = _safe_extract_json(raw)

    # validate structure and length
    if not isinstance(parsed, dict) or len(parsed) != n:
        raise ValueError(f"Generator returned invalid structure. Raw:\n{raw}")
    return parsed


# helpers to read/reset token counts
def get_token_count_record():
    global TOTAL_TOKEN_COUNT_EACH_GENERATION
    TOTAL_TOKEN_COUNT_EACH_GENERATION = np.append(TOTAL_TOKEN_COUNT_EACH_GENERATION, np.sum(TOTAL_TOKEN_COUNT))

    token_record = {
        'INPUT_token': np.sum(INPUT_TOKEN_COUNT),
        'OUTPUT_token': np.sum(OUTPUT_TOKEN_COUNT),
        'AVG_INPUT_token': np.average(INPUT_TOKEN_COUNT),
        'AVG_OUTPUT_token': np.average(OUTPUT_TOKEN_COUNT),
        'TOTAL_token': TOTAL_TOKEN_COUNT,
        f'TOTAL_token_for_{len(TOTAL_TOKEN_COUNT)}_mcqs' : TOTAL_TOKEN_COUNT_EACH_GENERATION,
        'AVG_TOTAL_token_PER_GENERATION': [np.average(TOTAL_TOKEN_COUNT_EACH_GENERATION), len(TOTAL_TOKEN_COUNT_EACH_GENERATION)],
    }

    return token_record


def reset_token_count(reset_all=None):
    """Call in app.py. For Reset Token Count after 1 Generation Session"""
    global INPUT_TOKEN_COUNT, OUTPUT_TOKEN_COUNT, TOTAL_TOKEN_COUNT, TOTAL_TOKEN_COUNT_EACH_GENERATION

    INPUT_TOKEN_COUNT = np.array([])
    OUTPUT_TOKEN_COUNT = np.array([])
    TOTAL_TOKEN_COUNT = np.array([])

    if reset_all:
        TOTAL_TOKEN_COUNT_EACH_GENERATION = np.array([])


def update_token_count(token_usage):
    """Update Token Count for each generation
    "usage": {
        "prompt_tokens": 1209,
        "completion_tokens": 313,
        "total_tokens": 1522,
        "prompt_tokens_details": {
        "cached_tokens": 0
    }
    """
    global INPUT_TOKEN_COUNT, OUTPUT_TOKEN_COUNT, TOTAL_TOKEN_COUNT # get value from global
    prompt_tokens = token_usage['prompt_tokens'] # INPUT token
    completion_tokens = token_usage['completion_tokens'] # OUTPUT token
    total_tokens = token_usage['total_tokens'] # TOTAL token

    INPUT_TOKEN_COUNT = np.append(INPUT_TOKEN_COUNT, prompt_tokens)
    OUTPUT_TOKEN_COUNT = np.append(OUTPUT_TOKEN_COUNT, completion_tokens)
    TOTAL_TOKEN_COUNT = np.append(TOTAL_TOKEN_COUNT, total_tokens)

    # print("Input Token Increase:", INPUT_TOKEN_COUNT)
    # print("Output Token Increase:", OUTPUT_TOKEN_COUNT)


def save_logs(record: dict, log_path:str = "logs/generation_log.jsonl"):
    """
    Append log to log_path
    record: dict with keys you want to store (e.g. filename, input/output token_count, collection, etc..)
    """
    # create file if not exist
    p = pathlib.Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # add id/timestampt if missing
    record.setdefault('id', str(uuid.uuid4()))
    record.setdefault('timestamp_utc', datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z") # get current time at timezone

    # append as 1 json file for each generation
    with open(p, "a", encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_time_info(time_info):
    """
    "time_info": {
        "queue_time": 0.000600429,
        "prompt_time": 0.052739054,
        "completion_time": 0.15692187,
        "total_time": 0.2117476463317871,
        "created": 1755599458
    }
    """
    time_info['created'] = time_info
    time_info['created'].pop('created')


def get_time_info():
    global TIME_INFOs
    return TIME_INFOs
    # token_record = {
    #     'completion_time': np.sum(INPUT_TOKEN_COUNT),
    #     'total_time': np.sum(OUTPUT_TOKEN_COUNT),
    # }

def log_pipeline(path, content):
    print("Save result to test/mcq_output.json")
    save_to_local(path=path, content=content)
    token_record = get_token_count_record()

    print("Token Record:")
    for record, value in token_record.items():
        print(f'{record}:{value}', '\n')

    reset_token_count()

def save_to_local(path, content):
    """
        path = 'test/raw_data.json'
        path = 'test/mcq_output.json'
        path = 'test/extract_output.md'

    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True) # create folder if missing
    p.touch(exist_ok=True) # create file if missing

    if path.lower().endswith('.json'):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(content, ensure_ascii=False, indent=2))
    else:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f'{content}') # md, txt

