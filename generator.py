import re
import random
import fitz
import numpy as np
import os
from typing import List, Optional, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
from uuid import uuid4
import pymupdf4llm

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        Distance,
        VectorParams,
    )
    from qdrant_client.http import models as rest
    _HAS_QDRANT = True
except Exception:
    _HAS_QDRANT = False

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

from utils import generate_mcqs_from_text, _post_chat, _safe_extract_json, save_to_local

class RAGMCQ:
    def __init__(
        self,
        embedder_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        generation_model: str = "gpt-oss-120b",
        qdrant_url: str = os.environ.get('QDRANT_URL') or "",
        qdrant_api_key: str = os.environ.get('QDRANT_API_KEY') or "",
        qdrant_prefer_grpc: bool = False,
    ):
        self.embedder = SentenceTransformer(embedder_model)
        self.generation_model = generation_model
        self.embeddings = None   # np.array of shape (N, D)
        self.texts = []          # list of chunk texts
        self.metadata = []       # list of dicts (page, chunk_id, char_range)
        self.index = None
        self.dim = self.embedder.get_sentence_embedding_dimension()

        self.qdrant = None
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.qdrant_prefer_grpc = qdrant_prefer_grpc

        if qdrant_url:
            self.connect_qdrant(qdrant_url, qdrant_api_key, qdrant_prefer_grpc)

    def extract_pages(
            self,
            pdf_path: str,
            *,
            pages: Optional[List[int]] = None,
            ignore_images: bool = False,
            dpi: int = 150
        ) -> List[str]:
            doc = fitz.open(pdf_path)
            try:
                # request page-wise output (page_chunks=True -> list[dict] per page)
                page_dicts = pymupdf4llm.to_markdown(
                    doc,
                    pages=pages,
                    ignore_images=ignore_images,
                    dpi=dpi,
                    page_chunks=True,
                )

                # to_markdown(..., page_chunks=True) returns a list of dicts, each has key "text" (markdown)
                pages_md: List[str] = []
                for p in page_dicts:
                    txt = p.get("text", "") or ""
                    pages_md.append(txt.strip())

                return pages_md
            finally:
                doc.close()

        # pages = []
        # with pdfplumber.open(pdf_path) as pdf:
        #     for p in pdf.pages:
        #         txt = p.extract_text() or ""
        #         pages.append(txt.strip())
        # return pages

    def chunk_text(self, text: str, max_chars: int = 1200, overlap: int = 100) -> List[str]:
        text = text.strip()
        if not text:
            return []

        if len(text) <= max_chars:
            return [text]

        # split by sentence-like boundaries
        sentences = re.split(r'(?<=[\.\?\!])\s+', text)
        chunks = []
        cur = ""

        for s in sentences:
            if len(cur) + len(s) + 1 <= max_chars:
                cur += (" " if cur else "") + s
            else:
                if cur:
                    chunks.append(cur)

                cur = (cur[-overlap:] + " " + s) if overlap > 0 else s

        if cur:
            chunks.append(cur)

        # if still too long, hard-split
        final = []
        for c in chunks:
            if len(c) <= max_chars:
                final.append(c)
            else:
                for i in range(0, len(c), max_chars):
                    final.append(c[i:i+max_chars])

        return final

    def build_index_from_pdf(self, pdf_path: str, max_chars: int = 1200):
        pages = self.extract_pages(pdf_path)

        self.texts = []
        self.metadata = []

        for p_idx, page_text in enumerate(pages, start=1):
            chunks = self.chunk_text(page_text or "", max_chars=max_chars)
            for cid, ch in enumerate(chunks, start=1):
                self.texts.append(ch)
                self.metadata.append({"page": p_idx, "chunk_id": cid, "length": len(ch)})

        if not self.texts:
            raise RuntimeError("No text extracted from PDF.")

        save_to_local('test/text_chunks.md', content=self.texts)

        # compute embeddings
        emb = self.embedder.encode(self.texts, convert_to_numpy=True, show_progress_bar=True)
        self.embeddings = emb.astype("float32")
        self._build_faiss_index()

    def _build_faiss_index(self, ef_construction=200, M=32):
        if _HAS_FAISS:
            d = self.embeddings.shape[1]
            index = faiss.IndexHNSWFlat(d, M)
            faiss.normalize_L2(self.embeddings)
            index.add(self.embeddings)
            index.hnsw.efConstruction = ef_construction
            self.index = index
        else:
            # store normalized embeddings and use brute-force numpy
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
            self.embeddings = self.embeddings / norms
            self.index = None

    def _retrieve(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")

        if _HAS_FAISS:
            faiss.normalize_L2(q_emb)
            D_list, I_list = self.index.search(q_emb, top_k)
            # D are inner products; return list of (idx, score)
            return [(int(i), float(d)) for i, d in zip(I_list[0], D_list[0]) if i != -1]
        else:
            qn = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
            sims = (self.embeddings @ qn.T).squeeze(axis=1)
            idxs = np.argsort(-sims)[:top_k]
            return [(int(i), float(sims[i])) for i in idxs]

    def generate_from_pdf(
        self,
        pdf_path: str,
        n_questions: int = 10,
        mode: str = "rag", # per_page or rag
        questions_per_page: int = 3, # for per_page mode
        top_k: int = 3, # chunks to retrieve for each question in rag mode
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        # build index
        self.build_index_from_pdf(pdf_path)

        output: Dict[str, Any] = {}
        qcount = 0

        if mode == "per_page":
            # iterate pages -> chunks
            for idx, meta in enumerate(self.metadata):
                chunk_text = self.texts[idx]

                if not chunk_text.strip():
                    continue
                to_gen = questions_per_page

                # ask generator
                try:
                    mcq_block = generate_mcqs_from_text(
                        chunk_text, n=to_gen, model=self.generation_model, temperature=temperature
                    )
                except Exception as e:
                    # skip this chunk if generator fails
                    print(f"Generator failed on page {meta['page']} chunk {meta['chunk_id']}: {e}")
                    continue

                for item in sorted(mcq_block.keys(), key=lambda x: int(x)):
                    qcount += 1
                    output[str(qcount)] = mcq_block[item]
                    if qcount >= n_questions:
                        return output

            return output

        elif mode == "rag":
            # strategy: create a few natural short queries by sampling sentences or using chunk summaries.
            # create queries by sampling chunk text sentences.
            # stop when n_questions reached or max_attempts exceeded.
            attempts = 0
            max_attempts = n_questions * 4

            while qcount < n_questions and attempts < max_attempts:
                attempts += 1
                # create a seed query: pick a random chunk, pick a sentence from it
                seed_idx = random.randrange(len(self.texts))
                chunk = self.texts[seed_idx]

                #? Investigate Chunking Strategy
                with open("chunks.txt", "a", encoding="utf-8") as f: f.write(chunk + "\n")

                sents = re.split(r'(?<=[\.\?\!])\s+', chunk)
                seed_sent = random.choice([s for s in sents if len(s.strip()) > 20]) if sents else chunk[:200]
                query = f"Create questions about: {seed_sent}"

                # retrieve top_k chunks
                retrieved = self._retrieve(query, top_k=top_k)
                context_parts = []
                for ridx, score in retrieved:
                    md = self.metadata[ridx]
                    context_parts.append(f"[page {md['page']}] {self.texts[ridx]}")
                context = "\n\n".join(context_parts)

                save_to_local('test/context.md', content=context)

                # call generator for 1 question (or small batch) with the retrieved context
                try:
                    # request 1 question at a time to keep diversity
                    mcq_block = generate_mcqs_from_text(
                        context, n=1, model=self.generation_model, temperature=temperature
                    )

                except Exception as e:
                    print(f"Generator failed during RAG attempt {attempts}: {e}")
                    continue

                # append result(s)
                for item in sorted(mcq_block.keys(), key=lambda x: int(x)):
                    qcount += 1
                    output[str(qcount)] = mcq_block[item]
                    if qcount >= n_questions:
                        return output

            return output
        else:
            raise ValueError("mode must be 'per_page' or 'rag'.")

    def validate_mcqs(
        self,
        mcqs: Dict[str, Any],
        top_k: int = 4,
        similarity_threshold: float = 0.5,
        evidence_score_cutoff: float = 0.5,
        use_model_verification: bool = True,
        model_verification_temperature: float = 0.0,
    ) -> Dict[str, Any]:
        if self.embeddings is None or not self.texts:
            raise RuntimeError("Index/embeddings not built. Run build_index_from_pdf() first.")

        report: Dict[str, Any] = {}

        # helper: semantic similarity search on statement -> returns list of (idx, score)
        def semantic_search(statement: str, k: int = top_k):
            q_emb = self.embedder.encode([statement], convert_to_numpy=True).astype("float32")

            if _HAS_FAISS:
                faiss.normalize_L2(q_emb)
                D_list, I_list = self.index.search(q_emb, k)
                # D are inner products; return list of (idx, score)
                return [(int(i), float(d)) for i, d in zip(I_list[0], D_list[0]) if i != -1]
            else:
                qn = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
                sims = (self.embeddings @ qn.T).squeeze(axis=1)
                idxs = np.argsort(-sims)[:k]
                return [(int(i), float(sims[i])) for i in idxs]

        # helper: verify with model (strict JSON in response)
        def _verify_with_model(question_text: str, options: Dict[str, str], correct_text: str, context_text: str):
            system = {
                "role": "system",
                "content": (
                    "Bạn là một trợ lý đánh giá tính thực chứng của câu hỏi trắc nghiệm dựa trên đoạn văn được cung cấp. Luôn trả lời bằng Tiếng Việt"
                    "Hãy trả lời DUY NHẤT bằng JSON hợp lệ (không có văn bản khác) theo schema:\n\n"
                    "{\n"
                    '  "supported": true/false,            # câu trả lời đúng có được nội dung chứng thực không\n'
                    '  "confidence": 0.0-1.0,              # mức độ tự tin (số)\n'
                    '  "evidence": "cụm văn bản ngắn làm bằng chứng hoặc trích dẫn",\n'
                    '  "reason": "ngắn gọn, vì sao supported hoặc không"\n'
                    "}\n\n"
                    "Luôn dựa chỉ trên nội dung trong trường 'Context' dưới đây. Nếu nội dung không chứa bằng chứng, trả về supported: false."
                )
            }
            user = {
                "role": "user",
                "content": (
                    "Câu hỏi:\n" + question_text + "\n\n"
                    "Lựa chọn:\n" + "\n".join([f"{k}: {v}" for k, v in options.items()]) + "\n\n"
                    "Đáp án:\n" + correct_text + "\n\n"
                    "Context:\n" + context_text + "\n\n"
                    "Hãy trả lời như yêu cầu."
                )
            }

            raw = _post_chat([system, user], model=self.generation_model, temperature=model_verification_temperature)

            # parse JSON object in response
            try:
                parsed = _safe_extract_json(raw)
            except Exception as e:
                return {"error": f"Model verification failed to return JSON: {e}", "raw": raw}
            return parsed

        # iterate MCQs
        for qid, item in mcqs.items():
            q_text = item.get("câu hỏi", "").strip()
            options = item.get("lựa chọn", {})
            correct_text = item.get("đáp án", "").strip()

            # form a short declarative statement to embed: "Question: ... Answer: <correct>"
            statement = f"{q_text} Answer: {correct_text}"

            retrieved = semantic_search(statement, k=top_k)
            evidence_list = []
            max_sim = 0.0
            for idx, score in retrieved:
                if score >= evidence_score_cutoff:
                    evidence_list.append({
                        "idx": idx,
                        "page": self.metadata[idx].get("page", None),
                        "score": float(score),
                        "text": (self.texts[idx][:1000] + ("..." if len(self.texts[idx]) > 1000 else "")),
                    })

                if score > max_sim:
                    max_sim = float(score)

            supported_by_embeddings = max_sim >= similarity_threshold

            model_verdict = None
            if use_model_verification:
                # build a context string from top retrieved chunks (regardless of cutoff)
                context_parts = []
                for ridx, sc in retrieved:
                    md = self.metadata[ridx]
                    context_parts.append(f"[page {md.get('page')}] {self.texts[ridx]}")
                context_text = "\n\n".join(context_parts)

                try:
                    parsed = _verify_with_model(q_text, options, correct_text, context_text)
                    model_verdict = parsed
                except Exception as e:
                    model_verdict = {"error": f"verification exception: {e}"}

            report[qid] = {
                "supported_by_embeddings": bool(supported_by_embeddings),
                "max_similarity": float(max_sim),
                "evidence": evidence_list,
                "model_verdict": model_verdict,
            }

        return report

    def connect_qdrant(self, url: str, api_key: str = None, prefer_grpc: bool = False):
        if not _HAS_QDRANT:
            raise RuntimeError("qdrant-client is not installed. Install with `pip install qdrant-client`.")
        self.qdrant_url = url
        self.qdrant_api_key = api_key
        self.qdrant_prefer_grpc = prefer_grpc
        # Create client
        self.qdrant = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)

    def _ensure_collection(self, collection_name: str):
        if self.qdrant is None:
            raise RuntimeError("Qdrant client not connected. Call connect_qdrant(...) first.")
        try:
            # get_collection will raise if not present
            _ = self.qdrant.get_collection(collection_name)
        except Exception:
            # create collection with vector size = self.dim
            vect_params = VectorParams(size=self.dim, distance=Distance.COSINE)
            self.qdrant.recreate_collection(collection_name=collection_name, vectors_config=vect_params)
            # recreate_collection ensures a clean collection; if you prefer to avoid wiping use create_collection instead.

    def save_pdf_to_qdrant(
        self,
        pdf_path: str,
        filename: str,
        collection: str,
        max_chars: int = 1200,
        batch_size: int = 64,
        overwrite: bool = False,
    ):
        if self.qdrant is None:
            raise RuntimeError("Qdrant client not connected. Call connect_qdrant(...) first.")

        # extract pages and chunks (re-using your existing helpers)
        pages = self.extract_pages(pdf_path)

        all_chunks = []
        all_meta = []
        for p_idx, page_text in enumerate(pages, start=1):
            chunks = self.chunk_text(page_text or "", max_chars=max_chars)
            for cid, ch in enumerate(chunks, start=1):
                all_chunks.append(ch)
                all_meta.append({"page": p_idx, "chunk_id": cid, "length": len(ch)})

        if not all_chunks:
            raise RuntimeError("No tSext extracted from PDF.")

        # ensure collection exists
        self._ensure_collection(collection)

        # optional: delete previous points for this filename if overwrite
        if overwrite:
            # delete by filter: filename == filename
            flt = Filter(must=[FieldCondition(key="filename", match=MatchValue(value=filename))])
            try:
                # qdrant-client delete uses delete(
                self.qdrant.delete(collection_name=collection, filter=flt)
            except Exception:
                # ignore if deletion fails
                pass

        # compute embeddings in batches
        embeddings = self.embedder.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True)
        embeddings = embeddings.astype("float32")

        # prepare points
        points = []
        for i, (emb, md, txt) in enumerate(zip(embeddings, all_meta, all_chunks)):
            pid = str(uuid4())
            source_id = f"{filename}__p{md['page']}__c{md['chunk_id']}"
            payload = {
                "filename": filename,
                "page": md["page"],
                "chunk_id": md["chunk_id"],
                "length": md["length"],
                "text": txt,
                "source_id": source_id,
            }
            points.append(PointStruct(id=pid, vector=emb.tolist(), payload=payload))

            # upsert in batches
            if len(points) >= batch_size:
                self.qdrant.upsert(collection_name=collection, points=points)
                points = []

        # upsert remaining
        if points:
            self.qdrant.upsert(collection_name=collection, points=points)

        try:
            self.qdrant.create_payload_index(
                collection_name=collection,
                field_name="filename",
                field_schema=rest.PayloadSchemaType.KEYWORD
            )
        except Exception as e:
            print(f"Index creation skipped or failed: {e}")

        return {"status": "ok", "uploaded_chunks": len(all_chunks), "collection": collection, "filename": filename}

    def list_files_in_collection(
        self,
        collection: str,
        payload_field: str = "filename",
        batch_size: int = 500,
    ) -> List[str]:
        if self.qdrant is None:
            raise RuntimeError("Qdrant client not connected. Call connect_qdrant(...) first.")

        # ensure collection exists
        try:
            if not self.qdrant.collection_exists(collection):
                raise RuntimeError(f"Collection '{collection}' does not exist.")
        except Exception:
            # collection_exists may raise if server unreachable
            raise

        filenames = set()
        offset = None

        while True:
            # scroll returns (points, next_offset)
            pts, next_offset = self.qdrant.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_payload=[payload_field],
                with_vectors=False,
            )

            if not pts:
                break

            for p in pts:
                # p may be a dict-like or an object with .payload
                payload = None
                if hasattr(p, "payload"):
                    payload = p.payload
                elif isinstance(p, dict):
                    # older/newer variants might use nested structures: try common keys
                    payload = p.get("payload") or p.get("payload", None) or p
                else:
                    # best-effort fallback: convert to dict if possible
                    try:
                        payload = dict(p)
                    except Exception:
                        payload = None

                if not payload:
                    continue

                # extract candidate value(s)
                val = None
                if isinstance(payload, dict):
                    val = payload.get(payload_field)
                else:
                    # Some payload representations store fields differently; try attribute access
                    val = getattr(payload, payload_field, None)

                # If value is list-like, iterate, else add single
                if isinstance(val, (list, tuple, set)):
                    for v in val:
                        if v is not None:
                            filenames.add(str(v))
                elif val is not None:
                    filenames.add(str(val))

            # stop if no more pages
            if not next_offset:
                break
            offset = next_offset

        return sorted(filenames)

    def list_chunks_for_filename(self, collection: str, filename: str, batch: int = 256) -> List[Dict[str, Any]]:
        if self.qdrant is None:
            raise RuntimeError("Qdrant client not connected. Call connect_qdrant(...) first.")

        results = []
        offset = None
        while True:
            # scroll returns (points, next_offset)
            points, next_offset = self.qdrant.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="filename", match=MatchValue(value=filename))
                    ]
                ),
                limit=batch,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            # points are objects (Record / ScoredPoint-like); get id and payload
            for p in points:
                # p.payload is a dict, p.id is point id
                results.append({"point_id": p.id, "payload": p.payload})
            if not next_offset:
                break
            offset = next_offset
        return results

    def _retrieve_qdrant(self, query: str, collection: str, filename: str = None, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        if self.qdrant is None:
            raise RuntimeError("Qdrant client not connected. Call connect_qdrant(...) first.")

        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")[0].tolist()
        q_filter = None
        if filename:
            q_filter = Filter(must=[FieldCondition(key="filename", match=MatchValue(value=filename))])

        search_res = self.qdrant.search(
            collection_name=collection,
            query_vector=q_emb,
            query_filter=q_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        out = []
        for hit in search_res:
            # hit.payload is the stored payload, hit.score is similarity
            out.append((hit.payload, float(getattr(hit, "score", 0.0))))
        return out

    def generate_from_qdrant(
        self,
        filename: str,
        collection: str,
        n_questions: int = 10,
        mode: str = "rag",               # 'per_chunk' or 'rag'
        questions_per_chunk: int = 3,    # used for 'per_chunk'
        top_k: int = 3,                  # retrieval size used in RAG
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        if self.qdrant is None:
            raise RuntimeError("Qdrant client not connected. Call connect_qdrant(...) first.")

        # get all chunks for this filename (payload should contain 'text', 'page', 'chunk_id', etc.)
        file_points = self.list_chunks_for_filename(collection=collection, filename=filename)
        if not file_points:
            raise RuntimeError(f"No chunks found for filename={filename} in collection={collection}.")

        # create a local list of texts & metadata for sampling
        texts = []
        metas = []
        for p in file_points:
            payload = p.get("payload", {})
            text = payload.get("text", "")
            texts.append(text)
            metas.append(payload)

        self.texts = texts
        self.metadata = metas
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        if embeddings is None or len(embeddings) == 0:
            self.embeddings = None
            self.index = None
        else:
            self.embeddings = embeddings.astype("float32")

            # update dim in case embedder changed unexpectedly
            self.dim = int(self.embeddings.shape[1])

            # build index
            self._build_faiss_index()

        output = {}
        qcount = 0

        if mode == "per_chunk":
            # iterate all chunks (in payload order) and request questions_per_chunk from each
            for i, txt in enumerate(texts):
                if not txt.strip():
                    continue
                to_gen = questions_per_chunk
                try:
                    mcq_block = generate_mcqs_from_text(txt, n=to_gen, model=self.generation_model, temperature=temperature)
                except Exception as e:
                    print(f"Generator failed on chunk (index {i}): {e}")
                    continue
                for item in sorted(mcq_block.keys(), key=lambda x: int(x)):
                    qcount += 1
                    output[str(qcount)] = mcq_block[item]
                    if qcount >= n_questions:
                        return output
            return output

        elif mode == "rag":
            attempts = 0
            max_attempts = n_questions * 4
            while qcount < n_questions and attempts < max_attempts:
                attempts += 1
                # create a seed query: pick a random chunk, pick a sentence from it
                seed_idx = random.randrange(len(self.texts))
                chunk = self.texts[seed_idx]
                sents = re.split(r'(?<=[\.\?\!])\s+', chunk)
                candidate = [s for s in sents if len(s.strip()) > 20]
                if candidate:
                    seed_sent = random.choice(candidate)
                else:
                    stripped = chunk.strip()
                    seed_sent = (stripped[:200] if stripped else "[no text available]")
                query = f"Create questions about: {seed_sent}"

                
                # retrieve top_k chunks from the same file (restricted by filename filter)
                retrieved = self._retrieve_qdrant(query=query, collection=collection, filename=filename, top_k=top_k)
                context_parts = []
                for payload, score in retrieved:
                    # payload should contain page & chunk_id and text
                    page = payload.get("page", "?")
                    ctxt = payload.get("text", "")
                    context_parts.append(f"[page {page}] {ctxt}")
                context = "\n\n".join(context_parts)

                try:
                    mcq_block = generate_mcqs_from_text(context, n=1, model=self.generation_model, temperature=temperature)
                except Exception as e:
                    print(f"Generator failed during RAG attempt {attempts}: {e}")
                    continue

                for item in sorted(mcq_block.keys(), key=lambda x: int(x)):
                    qcount += 1
                    output[str(qcount)] = mcq_block[item]
                    if qcount >= n_questions:
                        return output
            return output
        else:
            raise ValueError("mode must be 'per_chunk' or 'rag'.")
