import re
import json
from typing import Dict, Any
import requests
import os

#TODO: allow to choose different provider later + dynamic routing when token expired
API_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_API_KEY = os.environ['CEREBRAS_API_KEY']

HEADERS = {"Authorization": f"Bearer {CEREBRAS_API_KEY}"}
JSON_OBJ_RE = re.compile(r"(\{[\s\S]*\})", re.MULTILINE)

def _post_chat(messages: list, model: str, temperature: float = 0.2, timeout: int = 60) -> str:
    payload = {"model": model, "messages": messages, "temperature": temperature}
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

def generate_mcqs_from_text(
    source_text: str,
    n: int = 3,
    model: str = "gpt-oss-120b",
    temperature: float = 0.2,
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
            f"- Tạo đúng {n} mục, đánh YOUR_API_KEYsố từ 1 tới {n}.\n"
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
