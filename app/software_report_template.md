# Software Report: RAG-based MCQ Generation System

## 1. Overview / Abstract
The project provides an API service that ingests a PDF document and automatically generates multiple–choice questions (MCQs) using a Retrieval-Augmented Generation (RAG) pipeline. It exposes a FastAPI endpoint (`/generate`) that orchestrates: PDF text extraction → chunking → embedding + indexing → (mode-dependent) context selection → MCQ generation via an LLM (Together AI chat completion) → optional semantic + model-based validation.

Core components:
- Controller (FastAPI endpoints) – handles HTTP, file upload, response shaping.
- Use Case (RAGMCQ class) – encapsulates business logic: indexing, retrieval, generation, validation.
- Repositories / Data Stores – implicit: in‑memory lists of chunks, embeddings, optional FAISS index.

## 2. High-Level Workflow Diagram
### Mermaid Activity Diagram
```mermaid
flowchart LR
    A[Client Uploads PDF -> /generate] --> B{Mode?}
    B -->|rag| R1[Extract & Chunk PDF]
    B -->|per_page| R1
    R1 --> R2[SentenceTransformer Embeddings]
    R2 --> R3{FAISS Available?}
    R3 -->|Yes| R4[Build FAISS Index]
    R3 -->|No| R5[Normalize Embeddings (NumPy)]
    R4 --> R6[Question Generation Loop]
    R5 --> R6
    R6 -->|rag: sample queries + retrieve top-k| R7[Assemble Context]
    R6 -->|per_page: iterate chunks| R7
    R7 --> G1[Prompt LLM (JSON MCQs)]
    G1 --> P1[Parse & Validate JSON shape]
    P1 --> C{Need more?}
    C -->|Yes| R6
    C -->|No| V{Validation requested?}
    V -->|Yes| V1[Semantic Evidence Search + (Optional) Model Verification]
    V -->|No| OUT[Return MCQs]
    V1 --> OUT
```

### Alternative PlantUML Activity (Optional)
```plantuml
@startuml
start
:Upload PDF (multipart form);
:Select params (mode, n_questions,...);
:Extract pages via pdfplumber;
:Chunk text (sentence pack <= max_chars);
:Embed chunks (SentenceTransformer);
if (FAISS installed?) then (yes)
  :Build FAISS IndexFlatIP + L2 normalize;
else (no)
  :Keep normalized NumPy embeddings;
endif
repeat
if (mode == per_page) then (per_page)
  :Take next chunk;
else (rag)
  :Sample seed sentence;
  :Encode query & retrieve top-k chunks;
endif
:Assemble context;
:Call Together AI chat completion (prompt -> JSON);
:Parse JSON + accumulate MCQs;
repeat while (Need more questions?) is (yes)
end repeat
if (validate?) then (yes)
  :For each Q -> build statement;
  :Similarity search top_k evidence;
  if (Insufficient sim & model verify on) then (yes)
    :Call model for verification JSON;
  endif
  :Build validation report;
endif
:Return response JSON;
stop
@enduml
```

## 3. Repository–Controller–Use Case Abstraction
| Layer | Responsibility | In This Project |
|-------|---------------|-----------------|
| Controller | HTTP I/O, request validation, mapping domain results to API schema | `app.py` endpoints (`/health`, `/generate`) |
| Use Case | Orchestrates domain flow, independent of HTTP details | `RAGMCQ` methods: `build_index_from_pdf`, `generate_from_pdf`, `validate_mcqs` |
| Repository (implicit) | Data persistence / retrieval | In-memory: `texts`, `metadata`, `embeddings`, `FAISS index` (no external DB) |

Data Flow (simplified):
Client → Controller(`/generate`) → UseCase(`generate_from_pdf`) → (Extract + Chunk + Embed + Index + Retrieve + Generate) → Controller (normalize/optional validation) → Response

## 4. Detailed Pipeline Explanation
### 4.1 PDF Text Extraction & Chunking
- File saved to a temp path, then `pdfplumber` loads each page.
- `extract_pages()` returns list of raw page strings.
- `chunk_text()` packs sentences (regex split on punctuation boundaries) into segments up to `max_chars` (default 1200). If a sentence overflows, the existing chunk is flushed. Residual oversize chunks are hard-split.
- Metadata collected: page number, chunk id, length.

### 4.2 Embedding Generation
- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` loaded via `SentenceTransformer`.
- Batched encoding of all chunks → NumPy array (float32).
- If FAISS installed: L2 normalize embeddings, create `IndexFlatIP` (inner product ~ cosine after normalization), add embeddings.
- Else: manually store normalized embeddings for brute-force cosine similarity with matrix multiply.

### 4.3 Retrieval Strategy
Two modes:
1. `per_page`: Sequentially process each chunk; each call to the LLM asks for `questions_per_page` new MCQs until target `n_questions` reached.
2. `rag`: Loop builds a synthetic query by sampling a random chunk and a sentence. Retrieval:
   - Encode query → similarity search (FAISS or NumPy).
   - Take top-k chunk texts; join them with page tags as context.
   - Request 1 question per iteration (promotes diversity). Up to `max_attempts = n_questions * 4`.

Similarity Metric: Inner product on normalized vectors (equivalent to cosine). Sorting by descending similarity.

### 4.4 Question Generation Prompt Template
Implemented in `generate_mcqs_from_text` (utils):
- System message (Vietnamese) forcing strict JSON schema:
  ```json
  {
    "1": { "câu hỏi": "...", "lựa chọn": {"a":"...","b":"...","c":"...","d":"..."}, "đáp án":"..."},
    "2": { ... }
  }
  ```
- Constraints: exactly `n` entries; answer must be full text identical to one option; no explanations.
- User message: instructs generation from provided source text only.
- Post-processing: Regex extracts first JSON object; attempts `json.loads`; fallback removes trailing commas.

### 4.5 Validation (Optional)
For each MCQ (after normalization in controller):
1. Construct statement: `Question + Answer`.
2. Embed query → retrieve top_k evidence chunks.
3. Mark `supported_by_embeddings` if max similarity ≥ threshold.
4. If not supported and model verification enabled, call verification LLM prompt (also JSON-only) to assess `supported`, `confidence`, `evidence`, `reason`.

### 4.6 Together AI Integration
- Endpoint: `https://api.together.xyz/v1/chat/completions`.
- Authorization header uses `TOGETHER_KEY` environment variable.
- Payload: `{ model, messages, temperature }`.
- Response Handling: support both OpenAI-like `choices[0].message.content` and fallback `choices[0].text`.

## 5. API Endpoints
### 5.1 Health Check
GET `/health`
Response:
```json
{ "status": "ok", "ready": true }
```

### 5.2 Generate MCQs
POST `/generate` (multipart/form-data)
Fields:
- `file` (PDF) – required
- `n_questions` (int, default 10)
- `mode` ("rag" | "per_page", default "rag")
- `questions_per_page` (int, default 3) – used only in per_page mode
- `top_k` (int, default 3) – retrieval depth (rag & validation)
- `temperature` (float, default 0.2)
- `validate` (bool, default false)
- `debug` (bool) – if truthy writes `output.json` locally

Example Request (curl, PowerShell style quoting simplified):
```bash
curl -X POST http://localhost:8000/generate ^
  -F "file=@sample.pdf" ^
  -F "n_questions=5" ^
  -F "mode=rag" ^
  -F "top_k=3" ^
  -F "validate=true"
```

Success Response (validation on, abbreviated):
```json
{
  "mcqs": {
    "1": { "câu hỏi": "...", "lựa chọn": {"a":"...","b":"...","c":"...","d":"..."}, "đáp án": "..."},
    "2": { "câu hỏi": "...", "lựa chọn": { ... }, "đáp án": "..." }
  },
  "validation": {
    "1": {
      "supported_by_embeddings": true,
      "max_similarity": 0.83,
      "evidence": [ { "page": 2, "score": 0.81, "text": "Excerpt..." } ],
      "model_verdict": null
    }
  }
}
```

Error Examples:
- 400: non-PDF upload
- 500: generation pipeline error (e.g., empty PDF or model failure)
- 503: service not initialized

## 6. Data Structures & Types (Conceptual)
- Chunk: `{ text: str, page: int, chunk_id: int, length: int }`
- MCQ (generated raw): `{ "câu hỏi": str, "lựa chọn": {"a": str, ...}, "đáp án": str }`
- Normalized MCQ (API shaping): `{ mcq: str, options: { .. }, correct: str }`
- Validation Entry: `{ supported_by_embeddings: bool, max_similarity: float, evidence: [ {page, score, text}... ], model_verdict?: {...} }`

## 7. Configuration Points
| Parameter | Location | Purpose |
|-----------|----------|---------|
| `embedder_model` | `RAGMCQ.__init__` | Pretrained SentenceTransformer model name |
| `hf_model` | `RAGMCQ.__init__` | LLM model name for generation/verification |
| `top_k` | API form field & internal methods | Retrieval depth |
| `temperature` | API form field | Creativity vs determinism |
| `questions_per_page` | API form field | Batch size per chunk in per_page mode |

## 8. Simple Code Improvements (Quick Wins)
Below are low-risk refactors to make the code cleaner and more maintainable:

1. Environment Variable Safety:
   ```python
   def _require_env(name: str) -> str:
       val = os.getenv(name)
       if not val:
           raise RuntimeError(f"Missing required environment variable: {name}")
       return val
   TOGETHER_KEY = _require_env("TOGETHER_KEY")
   ```
2. Remove Unused Constant: `API_URL` in `utils.py` is unused (can delete to avoid confusion).
3. Unify Header Construction: Replace separate `HEADERS` / `TOGETHER_HEADERS` with a single function `auth_headers(provider)` that returns the correct dict.
4. Add Dataclass for MCQ:
   ```python
   from dataclasses import dataclass
   @dataclass
   class MCQ: question: str; options: Dict[str,str]; answer: str
   ```
   Helps type clarity in validation.
5. Extract Prompt Templates: Store system/user template strings as module-level constants to avoid duplication and ease future edits.
6. Fail-Fast on Empty PDF: Early check after extraction to return a user-friendly error message rather than a generic 500 later.
7. Replace Random Query Sampling Magic Numbers: Expose `max_attempts_factor` as a parameter (currently `n_questions * 4`).
8. Vector Normalization Consistency: Always keep an unnormalized copy if future scoring types are needed; currently normalization overwrites original when FAISS absent.
9. Logging Standardization: Replace scattered `print()` with Python `logging` module (configurable levels; avoids polluting stdout in production).
10. Validation Normalization: Move `_normalize_mcqs` from `app.py` into `RAGMCQ` (keeps domain logic together; controller stays thin).
11. Error Message Specificity: On generation failure wrap exceptions with context (page/chunk), but avoid leaking internal stack to clients; log full internally.
12. Dependency Pinning: Specify versions in `requirements.txt` for reproducibility (e.g., `sentence-transformers==2.2.2`).
13. Add `/models` Endpoint (Optional): Expose available embedder & generation models for UI introspection.
14. Add Basic Tests: e.g., a test for `chunk_text` (ensures boundaries) and JSON parsing fallback.
15. Reusable Retrieval: Expose a public `retrieve(query, top_k)` method to support future features (like user-specified queries) without duplicating private logic.

## 9. Potential Medium-Term Enhancements
| Area | Improvement |
|------|-------------|
| Prompt Robustness | Add JSON schema validation (e.g., `jsonschema`) & auto-regeneration for malformed outputs |
| Performance | Embed asynchronously / stream generation if backend supports it |
| Multi-Provider | Abstract provider strategy for HuggingFace, Together, OpenAI with pluggable client classes |
| Caching | Cache embeddings per PDF hash to avoid reprocessing identical documents |
| Analytics | Track generation latency, validation pass rate, average similarity in structured logs |
| i18n | Parameterize language; currently prompts in Vietnamese only |

## 10. Security & Operational Notes
- Ensure `TOGETHER_KEY` is not committed; rely on environment variables / secret managers.
- Limit PDF size and number of pages to prevent excessive memory or token usage.
- Consider sanitizing extracted text (remove personally identifiable info) before sending to LLM if sensitive documents are used.
- Add request timeout & retry logic for the LLM API (current single call may raise immediately).

## 11. Quick Start (Local)
1. Set API key: `setx TOGETHER_KEY "your_api_key"` (then restart shell).
2. Install dependencies: `pip install -r requirements.txt`.
3. Run API: `uvicorn app:app --reload`.
4. POST a PDF to `/generate`.

## 12. Summary
The system cleanly separates HTTP handling from the core RAG pipeline. Text is chunked at sentence boundaries, embedded, indexed (FAISS if available), and retrieved to assemble focused contexts that guide a JSON-constrained MCQ generation prompt. Optional validation uses embedding similarity and secondary model verification to flag unsupported questions. Suggested refactors improve safety, clarity, extensibility, and readiness for multi-provider expansion.

---
This report delivers architectural insight, workflow diagrams, detailed pipeline mechanics, API contract, and actionable improvement ideas for rapid comprehension and iteration.
