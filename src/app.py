import os
import shutil
import tempfile
from typing import List, Optional, Union

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the user's RAGMCQ implementation
from generator import RAGMCQWithDifficulty, RAGMCQ
from utils import log_pipeline

app = FastAPI(title="RAG MCQ Generator API")

# allow cross-origin requests (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# global rag instance
rag: Optional[RAGMCQ] = None
rag_difficulty: Optional[RAGMCQWithDifficulty] = None

class GenerateResponse(BaseModel):
    mcqs: dict
    validation: Optional[dict] = None

class ListResponse(BaseModel):
    files: list

@app.on_event("startup")
def startup_event():
    global rag_difficulty
    global rag

    # instantiate the heavy object once
    rag = RAGMCQ()
    rag_difficulty = RAGMCQWithDifficulty()
    print("RAGMCQ instance created on startup.")

@app.get("/health")
def health():
    return {"status": "ok", "ready": rag_difficulty is not None and rag is not None}

def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = ".pdf"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as out_file:
        shutil.copyfileobj(upload.file, out_file)
    return path


@app.get("/list_collection_files", response_model=ListResponse)
async def list_collection_files_endpoint(
    collection_name: str = "programming"
):
    global rag_difficulty
    if rag_difficulty is None:
        raise HTTPException(status_code=503, detail="RAGMCQ not ready on server.")

    files = rag_difficulty.list_files_in_collection(collection_name)

    return {"files": files}


@app.post("/upload_multiple_files", response_model=ListResponse)
async def upload_multiple_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...), # get multiple files
    collection_name: str = Form("programming"),
    overwrite: bool = Form(True),
    qdrant_filename_prefix: Optional[str] = Form(None),
):
    """
    Upload multiple PDF files and save their chunks to Qdrant.
    - files: one or more PDF files (multipart/form-data, repeated 'files' fields)
    - collection_name: Qdrant collection to save into
    - overwrite: if true, existing points for each filename will be removed
    - qdrant_filename_prefix: optional prefix; if provided each file will be saved under "<prefix>_<original_filename>"
    """
    global rag_difficulty
    if rag_difficulty is None:
        raise HTTPException(status_code=503, detail="RAGMCQ not ready on server.")

    saved_files = []

    def _cleanup(path: str):
        try:
            os.remove(path)
        except Exception:
            pass

    for idx, upload in enumerate(files):
        if isinstance(upload, str):
            continue

        if not upload.filename:
            raise HTTPException(status_code=400, detail=f"Uploaded file #{idx+1} missing filename.")

        if not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files supported: {upload.filename}, error at file number: {idx}")

        tmp_path = _save_upload_to_temp(upload)
        background_tasks.add_task(_cleanup, tmp_path)

        # decide filename to use in Qdrant payload
        qdrant_filename = str(
            f"{qdrant_filename_prefix}_{upload.filename}" if qdrant_filename_prefix else upload.filename
        )

        try:
            rag_difficulty.save_pdf_to_qdrant(tmp_path, filename=qdrant_filename, collection=collection_name, overwrite=overwrite)
            saved_files.append(qdrant_filename)
        except Exception as e:
            # collect failure info rather than aborting all uploads
            saved_files.append({"filename": upload.filename, "error": str(e)})

    return {"files": saved_files}



@app.post("/generate_saved_with_difficulty", response_model=GenerateResponse)
async def generate_saved_with_difficulty_endpoint(
    n_easy_questions: int = Form(3),
    n_medium_questions: int = Form(5), 
    n_hard_questions: int = Form(2),
    qdrant_filename: str = Form("default_filename"),
    collection_name: str = Form("programming"),
    mode: str = Form("rag"),
    questions_per_chunk: int = Form(5),
    top_k: int = Form(3),
    temperature: float = Form(0.2),
    validate_mcqs: bool = Form(False),
    enable_fiddler: bool = Form(False),
):
    global rag_difficulty
    if rag_difficulty is None:
        raise HTTPException(status_code=503, detail="RAGMCQ not ready on server.")
    
    difficulty_counts = {
        "easy": n_easy_questions,
        "medium": n_medium_questions,
        "hard": n_hard_questions
    }
    
    all_mcqs = {}
    counter = 1

    for difficulty, n_questions in difficulty_counts.items():
        try:
            mcqs = rag_difficulty.generate_from_qdrant(
                filename=qdrant_filename,
                collection=collection_name,
                n_questions=n_questions,
                mode=mode,
                questions_per_chunk=questions_per_chunk,
                top_k=top_k,
                temperature=temperature,
                enable_fiddler=enable_fiddler,
                target_difficulty=difficulty,
            )
            questions_list = []
            if isinstance(mcqs, dict):
                for v in mcqs.values():
                    if isinstance(v, list):
                        questions_list.extend(v)
                    else:
                        questions_list.append(v)
            elif isinstance(mcqs, list):
                questions_list = mcqs
            else:
                continue

            for qobj in questions_list:
                if isinstance(qobj, dict):
                    qobj["_difficulty"] = difficulty
                all_mcqs[str(counter)] = qobj
                counter += 1

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation from saved file failed: {e}")

    validation_report = None

    if validate_mcqs:
        try:
            # validate_mcqs expects keys as strings and the normalized content
            validation_report = rag_difficulty.validate_mcqs(all_mcqs, top_k=top_k)
        except Exception as e:
            # don't fail the whole request for a validation error — return generator output and note the error
            validation_report = {"error": f"Validation failed: {e}"}

    # log_pipeline('test/mcq_output.json', content={"mcqs": mcqs, "validation": validation_report})

    return {"mcqs": all_mcqs, "validation": validation_report}




@app.post("/generate_with_difficulty", response_model=GenerateResponse)
async def generate_with_difficulty_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    n_easy_questions: int = Form(3),
    n_medium_questions: int = Form(5), 
    n_hard_questions: int = Form(2),
    qdrant_filename: str = Form("default_filename"),
    collection_name: str = Form("programming"),
    mode: str = Form("rag"),
    questions_per_page: int = Form(5),
    top_k: int = Form(3),
    temperature: float = Form(0.2),
    validate_mcqs: bool = Form(False),
    enable_fiddler: bool = Form(False)
):
    global rag_difficulty
    if rag_difficulty is None:
        raise HTTPException(status_code=503, detail="RAGMCQ not ready on server.")

    # basic file validation
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # save uploaded file to a temp location
    tmp_path = _save_upload_to_temp(file)

    # ensure file removed afterward
    def _cleanup(path: str):
        try:
            os.remove(path)
        except Exception:
            pass

    background_tasks.add_task(_cleanup, tmp_path)

    # save pdf
    try:
        rag_difficulty.save_pdf_to_qdrant(tmp_path, filename=qdrant_filename, collection=collection_name, overwrite=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file to Qdrant Cloud: {e}")

    difficulty_counts = {
        "easy": n_easy_questions,
        "medium": n_medium_questions,
        "hard": n_hard_questions
    }
    
    all_mcqs = {}
    counter = 1

    for difficulty, n_questions in difficulty_counts.items():
        try:
            mcqs = rag_difficulty.generate_from_pdf(
                pdf_path=tmp_path,
                n_questions=n_questions,
                mode=mode,
                questions_per_page=questions_per_page,
                top_k=top_k,
                temperature=temperature,
                enable_fiddler=enable_fiddler,
                target_difficulty=difficulty,
            )
            questions_list = []
            if isinstance(mcqs, dict):
                for v in mcqs.values():
                    if isinstance(v, list):
                        questions_list.extend(v)
                    else:
                        questions_list.append(v)
            elif isinstance(mcqs, list):
                questions_list = mcqs
            else:
                continue

            for qobj in questions_list:
                if isinstance(qobj, dict):
                    qobj["_difficulty"] = difficulty
                all_mcqs[str(counter)] = qobj
                counter += 1

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation from file failed: {e}")

    validation_report = None

    if validate_mcqs:
        try:
            # rag.build_index_from_pdf(tmp_path)
            # validate_mcqs expects keys as strings and the normalized content
            validation_report = rag_difficulty.validate_mcqs(all_mcqs, top_k=top_k)
        except Exception as e:
            # don't fail the whole request for a validation error — return generator output and note the error
            validation_report = {"error": f"Validation failed: {e}"}


    # log_pipeline('test/mcq_output.json', content={"mcqs": mcqs, "validation": validation_report})

    return {"mcqs": all_mcqs, "validation": validation_report}


@app.post("/generate_saved", response_model=GenerateResponse)
async def generate_saved_endpoint(
    n_questions: int = Form(10),
    qdrant_filename: str = Form("default_filename"),
    collection_name: str = Form("programming"),
    mode: str = Form("rag"),
    questions_per_chunk: int = Form(5),
    top_k: int = Form(3),
    temperature: float = Form(0.2),
    validate_mcqs: bool = Form(False),
    enable_fiddler: bool = Form(False),
):
    global rag
    if rag is None:
        raise HTTPException(status_code=503, detail="RAGMCQ not ready on server.")
	
    try:
        mcqs = rag.generate_from_qdrant(
            filename=qdrant_filename,
            collection=collection_name,
            n_questions=n_questions,
            mode=mode,
            questions_per_chunk=questions_per_chunk,
            top_k=top_k,
            temperature=temperature,
            enable_fiddler=enable_fiddler
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation from saved file failed: {e}")

    validation_report = None

    if validate_mcqs:
        try:
            # validate_mcqs expects keys as strings and the normalized content
            validation_report = rag.validate_mcqs(mcqs, top_k=top_k)
        except Exception as e:
            # don't fail the whole request for a validation error — return generator output and note the error
            validation_report = {"error": f"Validation failed: {e}"}

    # log_pipeline('test/mcq_output.json', content={"mcqs": mcqs, "validation": validation_report})

    return {"mcqs": mcqs, "validation": validation_report}




@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    n_questions: int = Form(10),
    qdrant_filename: str = Form("default_filename"),
    collection_name: str = Form("programming"),
    mode: str = Form("rag"),
    questions_per_page: int = Form(5),
    top_k: int = Form(3),
    temperature: float = Form(0.2),
    validate_mcqs: bool = Form(False),
    enable_fiddler: bool = Form(False)
):
    global rag
    if rag is None:
        raise HTTPException(status_code=503, detail="RAGMCQ not ready on server.")

    # basic file validation
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # save uploaded file to a temp location
    tmp_path = _save_upload_to_temp(file)

    # ensure file removed afterward
    def _cleanup(path: str):
        try:
            os.remove(path)
        except Exception:
            pass

    background_tasks.add_task(_cleanup, tmp_path)

    # save pdf
    try:
        rag.save_pdf_to_qdrant(tmp_path, filename=qdrant_filename, collection=collection_name, overwrite=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file to Qdrant Cloud: {e}")

    # generate
    try:
        mcqs = rag.generate_from_pdf(
            tmp_path,
            n_questions=n_questions,
            mode=mode,
            questions_per_page=questions_per_page,
            top_k=top_k,
            temperature=temperature,
            enable_fiddler=enable_fiddler
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    validation_report = None

    if validate_mcqs:
        try:
            # rag.build_index_from_pdf(tmp_path)
            # validate_mcqs expects keys as strings and the normalized content
            validation_report = rag.validate_mcqs(mcqs, top_k=top_k)
        except Exception as e:
            # don't fail the whole request for a validation error — return generator output and note the error
            validation_report = {"error": f"Validation failed: {e}"}


    # log_pipeline('test/mcq_output.json', content={"mcqs": mcqs, "validation": validation_report})

    return {"mcqs": mcqs, "validation": validation_report}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
