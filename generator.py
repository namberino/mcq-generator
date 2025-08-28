import re
import random
import fitz
import string
import numpy as np
import os
from typing import List, Optional, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
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

from huggingface_hub import login
login(token=os.environ['HF_API_KEY'])

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
        self.qa_pipeline = pipeline("question-answering", model="nguyenvulebinh/vi-mrc-base", tokenizer="nguyenvulebinh/vi-mrc-base")
        self.cross_entail = CrossEncoder("itdainb/PhoRanker")
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

        # save_to_local('test/text_chunks.md', content=self.texts)

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

                #? investigate better Chunking Strategy
                #with open("chunks.txt", "a", encoding="utf-8") as f:
                    #f.write(chunk + "\n")

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

                # save_to_local('test/context.md', content=context)

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
        use_cross_encoder: bool = True,
        use_qa: bool = True,
        auto_accept_threshold: float = 0.7,
        review_threshold: float = 0.5,
        distractor_too_similar: float = 0.8,
        distractor_too_different: float = 0.15,
        model_verification_temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Upgraded validation pipeline:
            - embedding retrieval (self.index / self.embeddings)
            - cross-encoder entailment scoring (optional)
            - extractive QA consistency check (optional)
            - distractor similarity and type checks
            - aggregate into quality_score and triage_action

        Returns a dict keyed by qid with detailed info and triage decision.
        """
        cross_entail = None
        qa_pipeline = None
        if use_cross_encoder:
            try:
                cross_entail = self.cross_entail
            except Exception as e:
                cross_entail = None
        if use_qa:
            try:
                qa_pipeline = self.qa_pipeline
            except Exception:
                qa_pipeline = None

        # --- helpers ---
        def _norm_text(s: str) -> str:
            if s is None:
                return ""
            s = s.strip().lower()
            # remove punctuation
            s = s.translate(str.maketrans("", "", string.punctuation))
            # collapse whitespace
            s = " ".join(s.split())
            return s

        def _semantic_search(statement: str, k: int = top_k):
            # returns list of (idx, score) using current embeddings/index
            q_emb = self.embedder.encode([statement], convert_to_numpy=True).astype("float32")
            if _HAS_FAISS and getattr(self, "index", None) is not None:
                try:
                    faiss.normalize_L2(q_emb)
                    D_list, I_list = self.index.search(q_emb, k)
                    return [(int(i), float(d)) for i, d in zip(I_list[0], D_list[0]) if i != -1]
                except Exception:
                    pass
            # fallback to brute force
            qn = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
            sims = (self.embeddings @ qn.T).squeeze(axis=1)
            idxs = np.argsort(-sims)[:k]
            return [(int(i), float(sims[i])) for i in idxs]

        def _compose_context_from_retrieved(retrieved):
            parts = []
            for ridx, score in retrieved:
                md = self.metadata[ridx] if ridx < len(self.metadata) else {}
                page = md.get("page", "?")
                text = self.texts[ridx]
                parts.append(f"[page {page}] {text}")
            return "\n\n".join(parts)

        def _compute_option_embeddings(options_map: Dict[str, str]):
            # returns dict key->embedding
            keys = list(options_map.keys())
            texts = [options_map[k] for k in keys]
            embs = self.embedder.encode(texts, convert_to_numpy=True)
            return dict(zip(keys, embs))

        def _cosine(a, b):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
            return float(np.dot(a, b) / denom)

        # --- main loop ---
        report = {}
        for qid, item in mcqs.items():
            # support both Vietnamese keys and English keys
            q_text = (item.get("câu hỏi") or item.get("question") or item.get("q") or item.get("stem") or "").strip()
            options = item.get("lựa chọn") or item.get("options") or item.get("choices") or {}
            # options may be dict mapping letters to text, or list: normalize to dict
            if isinstance(options, list):
                options = {str(i+1): o for i, o in enumerate(options)}
            # correct answer may be a key (like "A") or the text; try both
            correct_key = item.get("đáp án") or item.get("answer") or item.get("correct") or item.get("ans")
            correct_text = ""
            if isinstance(correct_key, str) and correct_key.strip() in options:
                correct_text = options[correct_key.strip()]
            else:
                # maybe the answer is full text
                if isinstance(correct_key, str):
                    correct_text = correct_key.strip()
                else:
                    # fallback to 'correct_text' field
                    correct_text = item.get("correct_text") or item.get("đáp án_text") or ""

            # default empty guard
            options = {k: str(v) for k, v in options.items()}
            correct_text = str(correct_text)

            # prepare statement for retrieval
            statement = f"{q_text} Answer: {correct_text}"
            retrieved = _semantic_search(statement, k=top_k)
            # build context from top retrieved
            context_parts = []
            for ridx, score in retrieved:
                md = self.metadata[ridx] if ridx < len(self.metadata) else {}
                context_parts.append({"idx": ridx, "score": float(score), "page": md.get("page", None), "text": self.texts[ridx]})
            context_text = "\n\n".join([f"[page {p['page']}] {p['text']}" for p in context_parts])

            # Evidence list (embedding-based)
            evidence_list = []
            max_sim = 0.0
            for r in context_parts:
                if r["score"] >= evidence_score_cutoff:
                    snippet = r["text"]
                    evidence_list.append({
                        "idx": r["idx"],
                        "page": r["page"],
                        "score": r["score"],
                        "text": (snippet[:1000] + ("..." if len(snippet) > 1000 else "")),
                    })
                if r["score"] > max_sim:
                    max_sim = float(r["score"])
            supported_by_embeddings = max_sim >= similarity_threshold

            # Cross-encoder entailment scores for each option
            entailment_scores = {}
            correct_entail = 0.0
            try:
                if cross_entail is not None and context_text.strip():
                    # prepare list of (premise, hypothesis)
                    pairs = []
                    opt_keys = list(options.keys())
                    for k in opt_keys:
                        hyp = f"{q_text} Answer: {options[k]}"
                        pairs.append((context_text, hyp))
                    scores = cross_entail.predict(pairs)  # returns list of floats
                    # normalize scores to 0-1 if needed (cross-encoder may return arbitrary positive)
                    # do a min-max normalization across the returned scores
                    # but avoid division by zero
                    min_s = float(min(scores)) if len(scores) else 0.0
                    max_s = float(max(scores)) if len(scores) else 1.0
                    denom = max_s - min_s if max_s - min_s > 1e-6 else 1.0
                    for k, raw in zip(opt_keys, scores):
                        scaled = (raw - min_s) / denom
                        entailment_scores[k] = float(scaled)
                    # find correct key if available
                    # if `correct_text` exactly matches one of options, find that key
                    matched_key = None
                    for k, v in options.items():
                        if _norm_text(v) == _norm_text(correct_text):
                            matched_key = k
                            break
                    if matched_key:
                        correct_entail = entailment_scores.get(matched_key, 0.0)
                    else:
                        # fallback: treat 'correct_text' as a separate hypothesis
                        hyp = f"{q_text} Answer: {correct_text}"
                        raw = cross_entail.predict([(context_text, hyp)])[0]
                        # scale relative to min/max used above
                        correct_entail = float((raw - min_s) / denom)
                else:
                    entailment_scores = {}
                    correct_entail = 0.0
            except Exception as e:
                entailment_scores = {}
                correct_entail = 0.0

            def embed_cosine_sim(a, b):
                emb = self.embedder.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
                return float(np.dot(emb[0], emb[1]))

            # QA consistency
            qa_answer = None
            qa_score = 0.0
            qa_agrees = False
            if qa_pipeline is not None and context_text.strip():
                try:
                    qa_res = qa_pipeline(question=q_text, context=context_text)
                    # some QA pipelines return list of answers or dict
                    if isinstance(qa_res, list) and len(qa_res) > 0:
                        top = qa_res[0]
                        qa_answer = top.get("answer") if isinstance(top, dict) else str(top)
                        # qa_score = float(top.get("score", 0.0) if isinstance(top, dict) else 0.0)
                    elif isinstance(qa_res, dict):
                        qa_answer = qa_res.get("answer", "")
                        qa_score = float(qa_res.get("score", 0.0))
                    else:
                        qa_answer = str(qa_res)
                        qa_score = 0.0
                    qa_score = embed_cosine_sim(qa_answer, correct_text)
                    qa_agrees = (qa_score >= 0.5)
                except Exception:
                    qa_answer = None
                    qa_score = 0.0
                    qa_agrees = False

            try:
                opt_embs = _compute_option_embeddings({**options, "__CORRECT__": correct_text})
                correct_emb = opt_embs.pop("__CORRECT__")
                distractor_similarities = {}
                for k, emb in opt_embs.items():
                    distractor_similarities[k] = float(_cosine(correct_emb, emb))
            except Exception:
                distractor_similarities = {k: None for k in options.keys()}

            # distractor flags
            distractor_penalty = 0.0
            distractor_flags = []
            for k, sim in distractor_similarities.items():
                if sim is None or sim >= 0.999999 or (sim >= -0.01 and sim <= 0):
                    continue
                if sim >= distractor_too_similar:
                    distractor_flags.append({"key": k, "reason": "too_similar", "similarity": sim})
                    distractor_penalty += 0.25
                elif sim <= distractor_too_different:
                    distractor_flags.append({"key": k, "reason": "too_different", "similarity": sim})
                    distractor_penalty += 0.15
            # clamp penalty
            distractor_penalty = min(distractor_penalty, 1.0)

            # Ambiguity detection: how many options have entailment >= threshold
            ambiguous = False
            ambiguous_options = []
            if entailment_scores:
                # count options whose entailment >= max(correct_entail * 0.9, 0.6)
                amb_thresh = max(correct_entail * 0.9, 0.6)
                for k, sc in entailment_scores.items():
                    if sc >= amb_thresh and (options.get(k, "") != correct_text):
                        ambiguous_options.append({"key": k, "score": sc, "text": options[k]})
                ambiguous = len(ambiguous_options) > 0

            # Compose aggregated quality score
            # Components:
            #   - embedding_support: normalized max_sim (0..1)
            #   - entailment: correct_entail (0..1)
            #   - qa_agree: boolean -> 1 or 0 times qa_score
            #   - distractor_penalty: subtracted
            emb_support_norm = max_sim  # embedding similarity typically already 0..1 (inner product normalized)
            entail_component = float(correct_entail)
            qa_component = float(qa_score) if qa_agrees else 0.0

            # weighted sum
            quality_score = (
                0.40 * emb_support_norm +
                0.35 * entail_component +
                0.20 * qa_component -
                0.05 * distractor_penalty
            )
            # clamp to 0..1
            quality_score = max(0.0, min(1.0, quality_score))

            # triage decision
            triage_action = "reject"
            if quality_score >= auto_accept_threshold and not ambiguous:
                triage_action = "pass"
            elif quality_score >= review_threshold:
                triage_action = "review"
            else:
                triage_action = "reject"

            # compile flags/reasons
            flag_reasons = []
            if not supported_by_embeddings:
                flag_reasons.append("no_strong_embedding_evidence")
            if entailment_scores and correct_entail < 0.6:
                flag_reasons.append("low_entailment_score_for_correct")
            if qa_pipeline is not None and qa_score > 0.6 and not qa_agrees:
                flag_reasons.append("qa_contradiction")
            if ambiguous:
                flag_reasons.append("ambiguous_options_supported")
            if distractor_flags:
                flag_reasons.append({"distractor_issues": distractor_flags})

            # assemble per-question report
            report[qid] = {
                "supported_by_embeddings": bool(supported_by_embeddings),
                "max_similarity": float(max_sim),
                "evidence": evidence_list,
                "entailment_scores": entailment_scores,
                "correct_entailment": float(correct_entail),
                "qa_answer": qa_answer,
                "qa_score": float(qa_score),
                "qa_agrees": bool(qa_agrees),
                "distractor_similarities": distractor_similarities,
                "distractor_flags": distractor_flags,
                "distractor_penalty": float(distractor_penalty),
                "ambiguous_options": ambiguous_options,
                "quality_score": float(quality_score),
                "triage_action": triage_action,
                "flag_reasons": flag_reasons,
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
