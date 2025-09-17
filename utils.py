import re
import json
from typing import Dict, Any
import requests
import os
import numpy as np
import uuid
import datetime
import pathlib
import time

#TODO: allow to choose different provider later + dynamic routing when token expired
API_URL = "https://openrouter.ai/api/v1/chat/completions"
CEREBRAS_API_KEY = os.environ['OPENROUTER_KEY']

HEADERS = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
JSON_OBJ_RE = re.compile(r"(\{[\s\S]*\})", re.MULTILINE)

INPUT_TOKEN_COUNT = np.array([], dtype=int)
OUTPUT_TOKEN_COUNT = np.array([], dtype=int)
TOTAL_TOKEN_COUNT = np.array([], dtype=int)
TOTAL_TOKEN_COUNT_EACH_GENERATION = np.array([])
TIME_INFOs = {}


FIDDLER_GUARDRAILS_TOKEN = os.environ['FIDDLER_TOKEN']
SAFETY_GUARDRAILS_URL = "https://guardrails.cloud.fiddler.ai/v3/guardrails/ftl-safety"
GUARDRAILS_HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {FIDDLER_GUARDRAILS_TOKEN}',
}

def get_safety_response(text, sleep_seconds: float = 0.5):
    time.sleep(sleep_seconds) # rate limited
    response = requests.post(
        SAFETY_GUARDRAILS_URL,
        headers=GUARDRAILS_HEADERS,
        json={'data': {'input': text}},
    )
    response.raise_for_status()
    response_dict = response.json()
    return response_dict

def text_safety_check(text: str, sleep_seconds: float = 0.5):
    confs = get_safety_response(text, sleep_seconds)
    max_conf = max(confs.values())
    max_category = list(confs.keys())[list(confs.values()).index(max_conf)]
    return max_conf, max_category

def _post_chat(messages: list, model: str, temperature: float = 0.2, timeout: int = 60) -> str:
    payload = {"model": model, "messages": messages, "temperature": temperature, "provider": {"only": ["Cerebras", "together", "baseten", "deepinfra/fp4"]}}
    resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # handle various shapes
    if "choices" in data and len(data["choices"]) > 0:
        # prefer message.content
        ch = data["choices"][0]

        if isinstance(ch, dict) and "message" in ch and "content" in ch["message"]:
            return ch["message"]["content"]

        if "text" in ch:
            return ch["text"]

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


def structure_context_for_llm(
    source_text: str,
    model: str = "openai/gpt-oss-120b",
    temperature: float = 0.2,
    enable_fiddler = False,
) -> Dict[str, Any]:
    """
    Take a long source_text, split into N chunks, and restructure them
    so each chunk is self-contained, structured, and semantically meaningful.
    """

    system_message = {
        "role": "system",
        "content": (
            "Bạn là một trợ lý hữu ích chuyên xử lý và cấu trúc văn bản để phục vụ mô hình ngôn ngữ (LLM). Trả lời bằng Tiếng Việt\n"
            "Nhiệm vụ của bạn là:\n"
            "- Nếu văn bản dài trên 500 từ chia văn bản thành 2 đoạn (chunk) có ý nghĩa rõ ràng.\n"
            "- Mỗi chunk phải **tự chứa đủ thông tin** (self-contained) để LLM có thể hiểu độc lập.\n"
            "- Xác định **chủ đề chính (topic)** của mỗi chunk và dùng nó làm KEY trong JSON.\n"
            "- Trong mỗi topic, tổ chức thông tin thành cấu trúc rõ ràng gồm các trường:\n"
            "   - 'đoạn văn': nội dung gốc đã cấu trúc đầy đủ\n"
            "   - 'khái niệm chính': từ điểm chứa các khái niệm chính với khái niệm phụ hỗ trợ khái niệm chính đi kèm nếu có\n"
            "   - 'công thức': danh sách công thức (nếu có)\n"
            "   - 'ví dụ': ví dụ minh họa (nếu có)\n"
            "   - 'tóm tắt': tóm tắt nội dung, dễ hiểu\n"
            "- Giữ ngữ nghĩa liền mạch.\n"
            "- Chỉ TRẢ VỀ MỘT JSON hợp lệ theo schema, không kèm văn bản khác.\n\n"

            "Chỉ TRẢ VỀ duy nhất MỘT đối tượng JSON theo schema sau và không có bất kỳ văn bản nào khác:\n\n"
            "{\n"
            '  "Tên topic": {"đoạn văn": "nội dung đã cấu trúc của topic 1", "khái niệm chính": {"khái niệm chính 1":["khái niệm phụ", "..."],"khái niệm chính 2":["khái niệm phụ", "..."]}, "công thức": ["..."], "ví dụ": ["..."], "tóm tắt": "tóm tắt ngắn gọn"},\n'
            "}\n"
        )
    }

    user_message = {
        "role": "user",
        "content": (
            "Hãy chia văn bản sau thành nhiều chunk theo hướng dẫn trên và xuất JSON hợp lệ.\n"
            f"### Văn bản nguồn:\n{source_text}"
        )
    }

    if enable_fiddler:
        max_conf, max_cat = text_safety_check(user_message['content'])
        if max_conf > 0.5:
            print(f"Harmful content detected: ({max_cat} : {max_conf})")
            return {}

    raw = _post_chat([system_message, user_message], model=model, temperature=temperature)
    parsed = _safe_extract_json(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Generator returned invalid structure. Raw:\n{raw}")
    return parsed


def new_generate_mcqs_from_text(
    source_text: str,
    n: int = 3,
    model: str = "openai/openai/gpt-oss-120b",
    temperature: float = 0.2,
    enable_fiddler = False,
    target_difficulty: str = "easy",

) -> Dict[str, Any]:


    expected_concepts = {
      "easy": 1,
      "medium": 2,
      "hard": (3, 4)
    }
    if isinstance(expected_concepts[target_difficulty], tuple):
      min_concepts, max_concepts = expected_concepts[target_difficulty]
      concept_range = f"{min_concepts}-{max_concepts}"
    else:
      concept_range = expected_concepts[target_difficulty]


    difficulty_prompts = {
      "easy": (
          "- Câu hỏi DỄ: kiểm tra duy nhất 1 khái niệm chính cơ bản dễ hiểu, định nghĩa, hoặc công thức đơn giản."
          "- Đáp án có thể tìm thấy trực tiếp trong văn bản."
          "- Ngữ cảnh đủ để hiểu khái niệm chính."
          "- Distractors khác biệt rõ ràng, dễ loại bỏ."
          "- Độ dài câu hỏi ngắn gọn không quá 10-20 từ hoặc ít hơn 120 ký tự, tập trung vào một ý duy nhất.\n"
      ),
      "medium": (
          "- Câu hỏi TRUNG BÌNH kiểm tra khái niệm chính trong văn bản"
          "- Nếu câu hỏi thuộc dạng áp dụng và suy luận thiếu dữ liệu để trả lời câu hỏi, thêm nội dung hoặc ví dụ từ văn bản nguồn."
          "- Các Distractors không quá giống nhau."
          "- Độ dài câu hỏi vừa phải khoảng 23–30 từ hoặc khoảng 150 - 180 ký tự, có thêm chi tiết phụ để suy luận.\n"
      ),
      "hard": (
          "- Câu hỏi KHÓ kiểm tra thông tin được phân tích/tổng hợp"
          "- Nếu câu hỏi thuộc dạng áp dụng và suy luận thiếu dữ liệu để trả lời câu hỏi, thêm nội dung hoặc ví dụ từ văn bản nguồn."
          "- Ít nhất 2 distractors gần giống đáp án đúng, độ tương đồng cao. "
          f"- Đáp án yêu cầu học sinh suy luận hoặc áp dụng công thức vào ví dụ nếu có."
          "- Độ dài câu hỏi dài hơn 35 từ hoặc hơn 200 ký tự.\n \n"
      )
    }

    difficult_criteria = difficulty_prompts[target_difficulty] # "easy", "medium", "hard"
    print(concept_range)
    system_message = {
      "role": "system",
      "content": (
          "Bạn là một trợ lý hữu ích chuyên tạo câu hỏi trắc nghiệm (MCQ). Luôn trả lời bằng tiếng việt"
          f"Đảm bảo chỉ tạo sinh câu trắc nghiệm có độ khó sau {difficult_criteria}"
          f"Quan trọng: Mỗi câu hỏi chỉ sử dụng chính xác {concept_range} khái niệm chính (mỗi khái niệm chính có 1 danh sách khái niệm phụ) từ văn bản nguồn. "
          "Mỗi câu hỏi và đáp án phải dựa trên thông tin từ văn bản nguồn. Không được đưa kiến thức ngoài vào."
          "Chỉ TRẢ VỀ duy nhất một đối tượng JSON theo đúng schema sau và không kèm giải thích hay trường thêm:\n\n"
          "{\n"
          '  "1": { "câu hỏi": "...", "lựa chọn": {"a":"...","b":"...","c":"...","d":"..."}, "đáp án":"...", "khái niệm sử dụng": {"khái niệm chính":["khái niệm phụ", "..."], "..."]}},\n'
          '  "2": { ... }\n'
          "}\n\n"
          "Lưu ý:\n"
          f"- Tạo đúng {n} mục, đánh số từ 1 tới {n}.\n"
          "- Khóa 'lựa chọn' phải có các phím a, b, c, d.\n"
          "- 'đáp án' phải là toàn văn đáp án đúng (không phải ký tự chữ cái), và giá trị này phải khớp chính xác với một trong các giá trị trong 'options'.\n"
          "- Toàn bộ thông tin cần thiết để trả lời phải nằm trong chính câu hỏi, không tham chiếu lại văn bản nguồn."
          f"- Sử dụng chính xác {concept_range} khái niệm chính"
        )
    }

    user_message = {
        "role": "user",
        "content": (
            f"Hãy tạo {n} câu hỏi trắc nghiệm từ nội dung dưới đây. Chỉ sử dụng nội dung này làm nguồn duy nhất để xây dựng câu hỏi.\n\n"

            "### Yêu cầu:\n"
            "- Bám sát vào thông tin trong văn bản; không thêm kiến thức ngoài.\n"
            "- Nếu văn bản thiếu chi tiết, hãy tạo phương án nhiễu (distractors) hợp lý, nhưng phải có thể biện minh từ nội dung hoặc ngữ cảnh.\n"
            f"### Văn bản nguồn:\n{source_text}"
        )
    }


    if enable_fiddler:
        max_conf, max_cat = text_safety_check(user_message['content'])
        if max_conf > 0.5:
            print(f"Harmful content detected: ({max_cat} : {max_conf})")
            return {}

    raw = _post_chat([system_message, user_message], model=model, temperature=temperature)
    # print('\n\n',raw)
    parsed = _safe_extract_json(raw)
    # basic validation
    if not isinstance(parsed, dict) or len(parsed) != n:
        raise ValueError(f"Generator returned invalid structure. Raw:\n{raw}")
    return parsed



def generate_mcqs_from_text(
    source_text: str,
    n: int = 3,
    model: str = "openai/gpt-oss-120b",
    temperature: float = 0.2,
    enable_fiddler: bool = False,
) -> Dict[str, Any]:
    system_message = {
        "role": "system",
        "content": (
            "Bạn là một trợ lý hữu ích chuyên tạo câu hỏi trắc nghiệm. "
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

    if enable_fiddler:
        max_conf, max_cat = text_safety_check(user_message['content'])
        if max_conf > 0.5:
            print(f"Harmful content detected: ({max_cat} : {max_conf})")
            return {"error": "Harmful content detected", f"{max_cat}": f"{str(max_conf)}"}

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
        'INPUT_token_count': np.sum(INPUT_TOKEN_COUNT),
        'OUTPUT_token_count': np.sum(OUTPUT_TOKEN_COUNT),
        'AVG_INPUT_token_count': np.average(INPUT_TOKEN_COUNT),
        'AVG_OUTPUT_token_count': np.average(OUTPUT_TOKEN_COUNT),
        'TOTAL_token_count': TOTAL_TOKEN_COUNT,
        'TOTAL_token_count_PER_GENERATION - ': TOTAL_TOKEN_COUNT_EACH_GENERATION,
        'AVG_TOTAL_token_count_PER_GENERATION': [np.average(TOTAL_TOKEN_COUNT_EACH_GENERATION), len(TOTAL_TOKEN_COUNT_EACH_GENERATION)],
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
    #save_to_local(path=path, content=content)
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
