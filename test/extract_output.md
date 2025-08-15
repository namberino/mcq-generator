### AI VIET NAM – AI COURSE 2024

# Tutorial: RAG with LangChain for PDF Question Answering

### Dinh-Thang Duong, Nguyen-Thuan Duong, Quang-Vinh Dinh

# **I. Giới thiệu**

**LangChain** là một framework được thiết kếchuyên biệt cho việc triển khai LLMs trong các ứng
dụng thực tế. LangChain hỗtrợcác công cụvà thư viện mạnh mẽcho phép các nhà phát triển
dễdàng tích hợp các mô hình ngôn ngữlớn với các ứng dụng của họ, từcác Chatbot thông minh
cho đến các hệthống phân tích dữliệu phức tạp.


https://arxiv.org/pdf/1706.03762


Response















Hình 1: Minh họa ứng dụng hỏi đáp nội dung file pdf sửdụng LangChain.


Trong bài viết này, chúng ta sẽxây dựng một ứng dụng vềRAG (Retrieval Augmented Generation) trảlời các câu hỏi học thuật tận dụng nguồn tài liệu là các bài báo khoa học mà ta thu
thập được (dưới dạng file pdf), sửdụng thư viện LangChain. Tổng quan, pipeline của project
như sau:


1
**AI VIETNAM (AIO2024)** **aivietnam.edu.vn**























Hình 2: Tổng quan vềpipeline của project.


**Theo đó:**


1. Từdanh sách các bài báo khoa học, ta tách thành các văn bản nhỏ. Từđó, xây dựng một
hệcơ sởdữliệu vector với một embedding model.


2. Bên cạnh câu hỏi đầu vào (question), ta truy vấn các mẫu văn bản có liên quan đến đến
câu hỏi, dùng làm ngữcảnh (context) trong câu prompt. Đây là nguồn thông tin mà LLMs
có thểdựa vào đểtrảlời câu hỏi.


3. Đưa câu prompt vào mô hình (question và context) đểnhận câu trảlời từmô hình.


2
**AI VIETNAM (AIO2024)** **aivietnam.edu.vn**

# **II. Cài đặt chương trình**


Trong phần này, chúng ta sẽtiến hành cài đặt nội dung của project. Mã nguồn được xây dựng
trên hệđiều hành Ubuntu với GPU 24GB. Các bước thực hiện như sau:

## **II.1.** **Tổchức thư mục code**


Đểmã nguồn trởnên rõ ràng nhằm phục vụcho mục đích đọc hiểu code, chúng ta sẽtổchức
thư mục như sau:

```
               rag_langchain/

                data_source/

                  generative_ai/

                   download.py

                src/

                  base/

                   llm_model.py

                  rag/

                   file_loader.py

                   main.py

                   offline_rag.py

                   utils.py

                   vectorstore.py

                  app.py

                requirements.txt

```

Tổng quan, chúng ta sẽcó thư mục chứa mã nguồn có tên **rag_langchain** (các bạn hoàn toàn
có thểsửdụng tên gọi khác). Bên trong sẽcó các thư mục con và các file với ý nghĩa như sau:


_ **data_source/:** Thư mục dùng đểlưu trữcác tài liệu phục vụcho việc xây dựng hệcơ sở
dữliệu vector.


_ **data_source/generative_ai/download.py:** File code dùng đểtải tựđộng một sốcác
bài báo khoa học dưới dạng file pdf.


_ **src/base/llm_model:** File code dùng đểkhai báo hàm khởi tạo mô hình ngôn ngữlớn.


3
**AI VIETNAM (AIO2024)** **aivietnam.edu.vn**


_ **src/rag/:** Thư mục dùng đểlưu trữcác code liên quan đến xây dựng RAG, bao gồm:


1. **src/rag/file_loader.py:** File code dùng đểkhai báo các hàm load file pdf (vì tài
liệu của chúng ta thu thập thuộc file pdf).

2. **src/rag/main.py:** File code dùng đểkhai báo hàm khởi tạo chains.

3. **src/rag/offline_rag.py:** File code dùng đểkhai báo PromptTemplate.

4. **src/rag/utils.py:** File code dùng đểkhai báo hàm tách câu trảlời từmodel.

5. **src/rag/vectorstore.py:** File code dùng đểkhai báo hàm khởi tạo hệcơ sởdữliệu

vector.


_ **src/app.py:** File code dùng đểkhởi tạo API.


_ **requirements.txt:** File code dùng đểkhai báo các thư viện cần thiết đểsửdụng source
code.

## II.2. Cập nhật file requirements.txt


Đểbắt đầu, chúng ta sẽliệt kê các gói thư viện cần thiết đểchạy được chương trình này. Các
bạn hãy cập nhật file requirements.txt với nội dung sau:


1 `torch ==2.2.2`


2 `transformers ==4.39.3`


3 `accelerate ==0.28.0`


4 `bitsandbytes ==0.42.0`

5 `huggingface -hub==0.22.2`

6 `langchain ==0.1.14`

7 `langchain -core ==0.1.43`

8 `langchain -community ==0.0.31`

9 `pypdf ==4.2.0`


10 `sentence - transformers ==2.6.1`


11 `beautifulsoup4 ==4.12.3`

12 `langserve[` `all` `]`


13 `chromadb ==0.4.24`


14 `langchain -chroma ==0.1.0`

15 `faiss -cpu==1.8.0`

16 `rapidocr -onnxruntime ==1.3.16`


17 `unstructured ==0.13.2`


18 `fastapi ==0.110.1`


19 `uvicorn ==0.29.0`

## II.3. Cập nhật file data_source/generative_ai/download.- `py`


Đểtải một vài các bài báo khoa học làm dữliệu cho hệcơ sởdữliệu vector, chúng ta sẽxây
dựng một đoạn code tải tựđộng các bài báo. Nội dung như sau:


1 `import os`

2 `import` `wget`


4
**AI VIETNAM (AIO2024)** **aivietnam.edu.vn**


3


4 `file_links = [`


5 `{`


6 `"title"` `: "Attention Is All You Need"` `,`

7 `"url"` `: "https :// arxiv.org/pdf/1706.03762"`

8 `}` `,`

9 `{`


10 `"title"` `: "BERT - Pre -training of Deep` `Bidirectional` `Transformers` `for`
```
                          Language Understanding",
```

11 `"url"` `: "https :// arxiv.org/pdf/1810.04805"`

12 `}` `,`

13 `{`


14 `"title"` `: "Chain -of -Thought` `Prompting` `Elicits` `Reasoning in Large`
```
                          Language Models",
```

15 `"url"` `: "https :// arxiv.org/pdf/2201.11903"`

16 `}` `,`

17 `{`


18 `"title"` `: "Denoising` `Diffusion` `Probabilistic` `Models"` `,`

19 `"url"` `: "https :// arxiv.org/pdf/2006.11239"`

20 `}` `,`

21 `{`


22 `"title"` `: "Instruction` `Tuning for Large` `Language` `Models - A Survey"` `,`

23 `"url"` `: "https :// arxiv.org/pdf/2308.10792"`

24 `}` `,`

25 `{`


26 `"title"` `: "Llama 2- Open` `Foundation` `and Fine -Tuned` `Chat` `Models"` `,`

27 `"url"` `: "https :// arxiv.org/pdf/2307.09288"`

28 `}`


29 `]`


30


31 `def` `is_exist(file_link):`

32 `return os.path.exists(f` `"./` `{` `file_link[’title ’]` `}` `.pdf"` `)`


33


34 `for` `file_link in file_links:`


35 `if not` `is_exist(file_link):`

36 `wget.download(file_link[` `"url"` `], out=f` `"./` `{` `file_link[’title ’]` `}` `.pdf"` `)`


Trong file code trên, chúng ta cung cấp một list các đường dẫn bài báo. Từđó, sửdụng `wget` để
tải về. Các bài báo sẽđược lưu ngay tại vịtrí của file code. Vì mục đích demo, chúng ta sẽchỉ
tải một sốlượng nhỏcác paper. Các bạn có thểtựthêm vào nhiều paper khác đểtest.


5
**AI VIETNAM (AIO2024)** **aivietnam.edu.vn**


Hình 3: Minh họa danh sách các file bài báo khoa học sau khi được tải về.

## II.4. Cập nhật file src/base/llm_model.py


Tại file này, ta khai báo hàm `get_hf_model()`, dùng đểthực hiện tải và gọi pre-trained LLM
từHuggingFace vềmáy. Đồng thời, ta áp dụng kỹthuật quantization lên model đểthực hiện
inference trên GPU thấp. Nội dung file như sau:


1 `import` `torch`

2 `from` `transformers` `import` `BitsAndBytesConfig`

3 `from` `transformers` `import` `AutoTokenizer, AutoModelForCausalLM, pipeline`

4 `from` `langchain.llms. huggingface_pipeline` `import` `HuggingFacePipeline`


5


6


7 `nf4_config = BitsAndBytesConfig (`

8 `load_in_4bit=` `True,`


9 `bnb_4bit_quant_type =` `"nf4"` `,`

10 `bnb_4bit_use_double_quant =` `True,`

11 `bnb_4bit_compute_dtype =torch.bfloat16`

12 `)`


13


14 `def` `get_hf_llm(model_name: str = "meta -llama/Llama -3.2-3B -Instruct"` `,`

15 `max_new_token = 1024,`

16 `** kwargs):`


17


18 `model = AutoModelForCausalLM .from_pretrained (`

19 `model_name,`


20 `quantization_config =nf4_config,`

21 `low_cpu_mem_usage =` `True`

22 `)`

23 `tokenizer = AutoTokenizer. from_pretrained (model_name)`


6
**AI VIETNAM (AIO2024)** **aivietnam.edu.vn**


24


25 `model_pipeline = pipeline(`

26 `"text -generation"` `,`

27 `model=model,`


28 `tokenizer=tokenizer,`


29 `max_new_tokens =max_new_token,`


30 `pad_token_id=tokenizer.eos_token_id,`

31 `device_map=` `"auto"`

32 `)`


33


34 `llm = HuggingFacePipeline (`

35 `pipeline=model_pipeline,`

36 `model_kwargs=kwargs`

37 `)`


38


39 `return llm`


Trong project này, mô hình LLM mà chúng ta sửdụng là mô hình Llama 3.2 3B được huấn luyện
trên dữliệu instruction. Các bạn có thểthay thếbằng mô hình khác có cấu hình tương tự.

## II.5. Cập nhật file src/rag/file_loader.py


1 `from` `typing` `import Union, List, Literal`

2 `import` `glob`

3 `from tqdm` `import` `tqdm`

4 `import` `multiprocessing`

5 `from` `langchain_community . document_loaders` `import` `PyPDFLoader`

6 `from` `langchain_text_splitters` `import` `RecursiveCharacterTextSplitter`


7


8 `def` `remove_non_utf8_characters (text):`

9 `return ’’.join(char for char in text if ord` `(char) < 128)`


10


11 `def` `load_pdf(pdf_file):`

12 `docs = PyPDFLoader(pdf_file, extract_images=` `True` `).load ()`


13 `for doc in docs:`


14 `doc.page_content = remove_non_utf8_characters (doc.page_content)`


15 `return` `docs`


16


17 `def` `get_num_cpu ():`

18 `return` `multiprocessing.cpu_count ()`


19


20 `class` `BaseLoader:`


21 `def` `__init__(self) -> None` `:`

22 `self.num_processes = get_num_cpu ()`


23


24 `def` `__call__(self, files: List[` `str` `], ** kwargs):`


25 `pass`


26


27 `class` `PDFLoader(BaseLoader):`


28 `def` `__init__(self) -> None` `:`

29 `super ().__init__ ()`


30


31 `def` `__call__(self, pdf_files: List[` `str` `], ** kwargs):`

32 `num_processes = min` `(self.num_processes, kwargs[` `"workers"` `])`


7
**AI VIETNAM (AIO2024)** **aivietnam.edu.vn**


33 `with` `multiprocessing .Pool(processes=num_processes) as pool:`

34 `doc_loaded = []`

35 `total_files = len` `(pdf_files)`

36 `with tqdm(total=total_files, desc=` `"Loading` `PDFs"` `, unit=` `"file"` `) as`
```
                          pbar:
```

37 `for result in pool.imap_unordered (load_pdf, pdf_files):`

38 `doc_loaded.extend(result)`

39 `pbar.update(1)`


40 `return` `doc_loaded`


41


42 `class` `TextSplitter:`

43 `def` `__init__(self,`

44 `separators: List[` `str` `] = [` `’\n\n’` `, ’\n’` `, ’ ’` `, ’’],`

45 `chunk_size: int = 300,`


46 `chunk_overlap: int = 0`

47 `) -> None` `:`


48


49 `self.splitter = RecursiveCharacterTextSplitter (`


50 `separators=separators,`

51 `chunk_size=chunk_size,`


52 `chunk_overlap=chunk_overlap,`

53 `)`


54 `def` `__call__(self, documents):`

55 `return` `self.splitter.split_documents (documents)`


56


57 `class` `Loader:`


58 `def` `__init__(self,`

59 `file_type: str = Literal[` `"pdf"` `],`

60 `split_kwargs: dict = {`

61 `"chunk_size"` `: 300,`

62 `"chunk_overlap"` `: 0` `}`

63 `) -> None` `:`

64 `assert` `file_type in [` `"pdf"` `], "file_type` `must be pdf"`

65 `self.file_type = file_type`

66 `if file_type == "pdf"` `:`

67 `self.doc_loader = PDFLoader ()`


68 `else` `:`


69 `raise` `ValueError` `(` `"file_type` `must be pdf"` `)`


70


71 `self.doc_spltter = TextSplitter (** split_kwargs)`


72


73 `def load(self, pdf_files: Union[` `str, List[` `str` `]], workers: int = 1):`

74 `if isinstance` `(pdf_files, str` `):`

75 `pdf_files = [pdf_files]`

76 `doc_loaded = self.doc_loader(pdf_files, workers=workers)`

77 `doc_split = self.doc_spltter(doc_loaded)`

78 `return` `doc_split`


79


80 `def` `load_dir(self, dir_path: str, workers: int = 1):`

81 `if self.file_type == "pdf"` `:`

82 `files = glob.glob(f` `"` `{` `dir_path` `}` `/*. pdf"` `)`

83 `assert len` `(files) > 0, f` `"No {` `self.file_type` `} files` `found in {`
```
                          dir_path } "

```

84 `else` `:`


85 `raise` `ValueError` `(` `"file_type` `must be pdf"` `)`


8
**AI VIETNAM (AIO2024)** **aivietnam.edu.vn**


86 `return` `self.load(files, workers=workers)`

## II.6. Cập nhật file src/rag/vectorstore.py


Tại file này, ta định nghĩa một class đểkhởi tạo hệcơ sởdữliệu vector. Trong project này, chúng
ta sẽsửdụng Chroma. Vềviệc tìm kiếm tài liệu tương đồng, ta sửdụng FAISS. Như vậy, nội
dung của file như sau:


Hình 4: Minh họa việc sửdụng vector database Chroma đểtruy vấn các tài liệu có liên quan
[làm context trong prompt. Ảnh: Link.](https://heidloff.net/article/retrieval-augmented-generation-chroma-langchain/)


1 `from` `typing` `import` `Union`

2 `from` `langchain_chroma` `import` `Chroma`

3 `from` `langchain_community .vectorstores` `import` `FAISS`

4 `from` `langchain_community .embeddings` `import` `HuggingFaceEmbeddings`


5


6 `class` `VectorDB:`


7 `def` `__init__(self,`


8 `documents = None,`

9 `vector_db: Union[Chroma, FAISS] = Chroma,`

10 `embedding = HuggingFaceEmbeddings (),`

11 `) -> None` `:`


12


13 `self.vector_db = vector_db`


14 `self.embedding = embedding`

15 `self.db = self._build_db(documents)`


16


9
**AI VIETNAM (AIO2024)** **aivietnam.edu.vn**


17 `def` `_build_db(self, documents):`

18 `db = self.vector_db. from_documents (documents=documents,`

19 `embedding=self.embedding)`


20 `return db`


21


22 `def` `get_retriever(self,`

23 `search_type: str = "similarity"` `,`

24 `search_kwargs: dict = {` `"k"` `: 10` `}`

25 `):`

26 `retriever = self.db.as_retriever(search_type=search_type,`

27 `search_kwargs=search_kwargs)`


28 `return` `retriever`

## II.7. Cập nhật file src/rag/offline_rag.py


Tại file này, ta khai báo class `Offline_RAG` đểxây dựng một chain vềRAG, bao gồm việc sử
dụng retriever lấy context, xây dựng prompt và đưa vào model. Nội dung của file như sau:


1 `import re`

2 `from` `langchain` `import hub`

3 `from` `langchain_core .runnables` `import` `RunnablePassthrough`

4 `from` `langchain_core .output_parsers` `import` `StrOutputParser`


5


6 `class` `Str_OutputParser ( StrOutputParser ):`

7 `def` `__init__(self) -> None` `:`

8 `super ().__init__ ()`


9


10 `def parse(self, text: str` `) -> str` `:`

11 `return` `self.extract_answer(text)`


12


13 `def` `extract_answer (self,`


14 `text_response: str,`

15 `pattern: str = r"Answer :\s*(.*)"`

16 `) -> str` `:`


17


18 `match = re.search(pattern, text_response, re.DOTALL)`


19 `if match:`


20 `answer_text = match.group(1).strip ()`


21 `return` `answer_text`


22 `else` `:`


23 `return` `text_response`


24


25 `class` `Offline_RAG:`

26 `def` `__init__(self, llm) -> None` `:`


27 `self.llm = llm`


28 `self.prompt = hub.pull(` `"rlm/rag -prompt"` `)`

29 `self.str_parser = Str_OutputParser ()`


30


31 `def` `get_chain(self, retriever):`

32 `input_data = {`

33 `"context"` `: retriever | self.format_docs,`

34 `"question"` `: RunnablePassthrough ()`

35 `}`

36 `rag_chain = (`


10
**AI VIETNAM (AIO2024)** **aivietnam.edu.vn**


37 `input_data`

38 `| self.prompt`

39 `| self.llm`

40 `| self.str_parser`

41 `)`


42 `return` `rag_chain`


43


44 `def` `format_docs(self, docs):`

45 `return "\n\n"` `.join(doc.page_content` `for doc in docs)`

## II.8. Cập nhật file src/rag/utils.py


Tại file này, ta khai báo hàm tách phần trảlời của model từcâu prompt (phần bắt đầu từ
**“Answer:”** ):


1 `import re`


2


3 `def` `extract_answer (text_response: str,`

4 `pattern: str = r"Answer :\s*(.*)"`

5 `) -> str` `:`


6


7 `match = re.search(pattern, text_response)`


8 `if match:`


9 `answer_text = match.group(1).strip ()`


10 `return` `answer_text`


11 `else` `:`


12 `return "Answer not found."`

## II.9. Cập nhật file src/rag/main.py


Tại file này, ta khởi tạo toàn bộcác instance của các class, các hàm mà ta đã khai báo trước đó
và kết nối chúng vào trong một hàm duy nhất gọi là `build_rag_chain()` :


1 `from` `pydantic` `import` `BaseModel, Field`


2


3 `from src.rag.file_loader` `import` `Loader`

4 `from src.rag.vectorstore` `import` `VectorDB`

5 `from src.rag.offline_rag` `import` `Offline_RAG`


6


7 `class` `InputQA(BaseModel):`

8 `question: str = Field (..., title=` `"Question to ask the model"` `)`


9


10 `class` `OutputQA(BaseModel):`

11 `answer: str = Field (..., title=` `"Answer` `from the model"` `)`


12


13 `def` `build_rag_chain (llm, data_dir, data_type):`

14 `doc_loaded = Loader(file_type=data_type).load_dir(data_dir, workers=2)`

15 `retriever = VectorDB(documents = doc_loaded).get_retriever ()`

16 `rag_chain = Offline_RAG(llm).get_chain(retriever)`


17


18 `return` `rag_chain`


11
**AI VIETNAM (AIO2024)** **aivietnam.edu.vn**


Như vậy, ta đã hoàn thiện toàn bộcác code cần thiết đểxây dựng một ứng dụng vềRAG. Để
tổng quát hóa toàn bộquy trình, chúng ta có thểtham khảo qua ảnh sau:


[Link.](https://medium.com/@akriti.upadhyay/implementing-rag-with-langchain-and-hugging-face-28e3ea66c5f7)
Hình 5: Minh họa chuỗi (chain) các bước xây dựng RAG trong LangChain. Ảnh:

## II.10. Cập nhật file src/app.py


Cuối cùng, ta tạo file dùng đểkhai báo API với LangServe đểtriển khai ứng dụng RAG. Đối
với LangServe, cách sửdụng gần như tương tựvới việc sửdụng FastAPI. Nội dung file code như

sau:


1 `import os`

2 `os.environ[` `" TOKENIZERS_PARALLELISM "` `] = "false"`


3


4 `from` `fastapi` `import` `FastAPI`

5 `from` `fastapi.middleware.cors` `import` `CORSMiddleware`


6


7 `from` `langserve` `import` `add_routes`


8


9 `from src.base.llm_model` `import` `get_hf_llm`

10 `from src.rag.main` `import` `build_rag_chain, InputQA, OutputQA`


11


12 `llm = get_hf_llm(temperature=0.9)`

13 `genai_docs = "./ data_source/generative_ai"`


14


15 _`# --------- Chains ----------------`_


16


17 `genai_chain = build_rag_chain (llm, data_dir=genai_docs, data_type=` `"pdf"` `)`


18


19 _`# --------- App - FastAPI`_ _`----------------`_


20


21 `app = FastAPI(`

22 `title=` `"LangChain` `Server"` `,`

23 `version=` `"1.0"` `,`


24 `description=` `"A simple api server` `using` `Langchain ’s Runnable` `interfaces"` `,`

25 `)`


26


27 `app. add_middleware (`

28 `CORSMiddleware,`

29 `allow_origins =[` `"*"` `],`


12
**AI VIETNAM (AIO2024)** **aivietnam.edu.vn**


30 `allow_credentials =` `True,`

31 `allow_methods =[` `"*"` `],`

32 `allow_headers =[` `"*"` `],`

33 `expose_headers =[` `"*"` `],`

34 `)`


35


36 _`# --------- Routes - FastAPI`_ _`----------------`_


37


38 `@app.get(` `"/check"` `)`

39 `async def check ():`

40 `return {` `"status"` `: "ok"` `}`


41


42 `@app.post(` `"/generative_ai"` `, response_model =OutputQA)`

43 `async def` `generative_ai(inputs: InputQA):`

44 `answer = genai_chain.invoke(inputs.question)`

45 `return {` `"answer"` `: answer` `}`


46


47 _`# --------- Langserve`_ _`Routes - Playground`_ _`----------------`_

48 `add_routes(app,`

49 `genai_chain,`

50 `playground_type=` `"default"` `,`

51 `path=` `"/generative_ai"` `)`


Đểkhởi động API, chúng ta duy chuyển đến thư mục root của source code trong terminal (trong
trường hợp của bài viết sẽlà thư mục `rag_langchain/` ), sửdụng lệnh sau (sau khi đã cài đặt
các thư viện cần thiết cũng như vector database). Lưu ý, nếu bịlỗi do port đã được sửdụng
trong máy của bạn thì có thểthay đổi sang một port khác:


1 `uvicorn src.app:app --host "0.0.0.0" --port 5000 --` `reload`


13
**AI VIETNAM (AIO2024)** **aivietnam.edu.vn**


Hình 6: Minh họa API sau khi ta triển khai thành công.


Hình 7: Minh họa một kết quảcủa model thông qua API mà chúng ta đã xây dựng.


14
**AI VIETNAM (AIO2024)** **aivietnam.edu.vn**

# **III.** **Câu hỏi trắc nghiệm**


1. LangChain được sửdụng nhằm mục đích gì?


( _a_ ) Web Scraping.


( _b_ ) Model Quantization.


( _c_ ) Building language model-powered applications.


( _d_ ) Database Management.


2. Nội dung nào dưới đây là một thành phần cốt lõi của LangChain?


( _a_ ) Transformers.


( _b_ ) Agents.


( _c_ ) Callbacks.


( _d_ ) Hooks.


3. Trong LangChain, mục đích trong việc sửdụng PromptTemplate là?


( _a_ ) Tạo các trường thông tin trong hệcơ sởdữliệu lưu thông tin người dùng.


( _b_ ) Định nghĩa các tính năng trong giao diện của người dùng.


( _c_ ) Tối ưu tốc độxửlý của mô hình.

( _d_ ) Chuẩn hóa một cấu trúc phản hồi nhất quán từmô hình.


4. Xét đoạn code dưới đây:


1 `from` `langchain_openai` `import` `ChatOpenAI`

2 `from` `langchain_openai` `import` `OpenAI`


3


4 `llm = OpenAI ()`

5 `chat_model = ChatOpenAI(model=` `"gpt -3.5-turbo -0125"` `)`


Ý nghĩa của đoạn code trên là?


( _a_ ) Khởi tạo model GPT 3.5 Turbo-0125.


( _b_ ) Tải pre-trained model GPT 3.5 Turbo-0125.

( _c_ ) Kiểm tra tốc độđường truyền với ChatGPT API.


( _d_ ) Các đáp án trên đều sai.


5. Ý nghĩa của phương thức `from_template()` trong class PromptTemplate là?


( _a_ ) Đểkhởi tạo prompt template từmột file.

( _b_ ) Đểkhởi tạo prompt template từmột string.

( _c_ ) Đểkhởi tạo prompt template từmột danh sách các tin nhắn.

( _d_ ) Đểkhởi tạo prompt template từmột prompt template có sẵn.


15
**AI VIETNAM (AIO2024)** **aivietnam.edu.vn**


6. Trong LangChain, loại OutputParser nào dưới đây có thểđược sửdụng đểtrảvềkết quả
của mô hình dưới dạng JSON?


( _a_ ) PydanticOutputParser.


( _b_ ) RegexOutputParser.


( _c_ ) JsonOutputParser.


( _d_ ) YamlOutputParser.


7. Xét đoạn code dưới đây:


1 `from` `langchain` `import` `HuggingFaceHub`

2 `from` `langchain` `import` `PromptTemplate`


3


4 `template =` _`""" Question:`_ `{` _`question`_ `}`


5


6 _`Answer: """`_


7 `prompt = PromptTemplate(`

8 `template=template,`

9 `input_variables =[` `’question ’]`

10 `)`


11


12 `hub_llm = HuggingFaceHub(`

13 `repo_id=` `’google/flan -t5 -xl’`

14 `)`


15


16 `llm_chain = prompt | hub_llm`


17


18 `print` `(llm_chain.run(` `"What year was the World Cup first` `held?"` `))`


Ý nghĩa của các dòng code 16 là gì?


( _a_ ) Khai báo hệcơ sởdữliệu vector.


( _b_ ) Khởi tạo LLMChain với LLM và Prompt.


( _c_ ) Cài đặt ủy quyền và bảo mật cho người dùng.


( _d_ ) Phân tích và trực quan hóa dữliệu.


8. Xét đoạn code dưới đây:


1 `from` `langchain_community . document_loaders` `import` `PyPDFLoader`


2


3 `pdf_loader = PyPDFLoader(url, extract_images =` `True` `)`


4


5 `docs = pdf_loader.load ()`


Tham số `extract_images` tại dòng code 3 có chức năng gì?


( _a_ ) Trảvềtất cảảnh từfile pdf.


( _b_ ) Bỏqua ảnh, chỉload text.


( _c_ ) Phân tích ảnh thành vector.

( _d_ ) Chuyển đổi ảnh trong file pdf thành text.


16
**AI VIETNAM (AIO2024)** **aivietnam.edu.vn**


9. Tại sao chúng ta cần phải chia nhỏcác tài liệu đầu vào thành các tài liệu ngắn hơn? Chọn
câu trảlời **SAI** .


( _a_ ) Giúp LLM tập trung tạo ra câu trảlời chỉdựa trên các thông tin có liên quan.


( _b_ ) Tiết kiệm bộnhớcho phần cứng.


( _c_ ) Chỉdựa vào một phần nhỏtài liệu thì mô hình vẫn trảlời chính xác.


( _d_ ) Giúp mô hình LLM chạy nhanh hơn.


10. Xét đoạn code dưới đây:


1 `from` `langchain_community . document_loaders` `import` `PyPDFLoader`

2 `from` `langchain_text_splitters` `import` `RecursiveCharacterTextSplitter`

3 `from` `langchain_community .embeddings` `import` `HuggingFaceEmbeddings`

4 `from` `langchain_chroma` `import` `Chroma`


5


6 `pdf_url = "https :// arxiv.org/pdf/2401.18059v1.pdf"`


7


8 _`# PDF loader`_


9 `pdf_loader = PyPDFLoader(pdf_url, extract_images =` `True` `)`

10 `pdf_pages = pdf_loader.load ()`


11


12 _`# Splitter`_

13 `splitter = RecursiveCharacterTextSplitter (`

14 `chunk_size=300,`


15 `chunk_overlap=0,`

16 `)`

17 `docs = splitter. split_documents (pdf_pages)`


18


19 _`# Embedding`_ _`model`_

20 `embedding_model = HuggingFaceEmbeddings ()`


21


22 _`# vector`_ _`store`_


23 `chroma_db = Chroma.from_documents(docs, embedding= embedding_model )`


Nhiệm vụcủa `embedding_model` là gì?


( _a_ ) Dùng biến đổi chuỗi đầu vào thành các vector cho cơ sởdữliệu vector.

( _b_ ) Dùng đểlập chỉmục cho cơ sởdữliệu.

( _c_ ) Dùng đểtìm kiếm tài liệu.

( _d_ ) Dùng đểtính toán độtương đồng.


17
**AI VIETNAM (AIO2024)** **aivietnam.edu.vn**

# **IV. Phụlục**


1. **Datasets:** [Các file dataset được đềcập trong bài có thểđược tải tại đây.](https://drive.google.com/drive/folders/1q5klM8jTCa0VpJ8PIg6vtf0znCNWKysY?usp=drive_link)


2. **Hint:** [Các file code gợi ý có thểđược tải tại đây.](https://drive.google.com/drive/folders/1X2tjSU9W5SJgucN1GyqaOU0U4-hJnFiN?usp=drive_link)


3. **Solution:** Các file code cài đặt hoàn chỉnh và phần trảlời nội dung trắc nghiệm có thể
[được tải tại đây (Lưu ý: Sáng thứ3 khi hết deadline phần nội dung này, ad mới copy các](https://drive.google.com/drive/folders/1ZSwNOyYU8EWmpJasGamse_ObqgaTmB8U?usp=drive_link)
tài liệu bài giải nêu trên vào đường dẫn).


4. **Demo:** [Web demo và mã nguồn của ứng dụng có thểđược truy cập tại đây.](https://huggingface.co/spaces/VLAI-AIVN/AIO2024M10_RAG_Langchain)


5. **Rubric:**



|Mục|Kiến Thức|Đánh Giá|
|---|---|---|
|I.|- Kiến thức vềmô hình ngôn ngữlớn<br>(LLMs).<br>- Kiến thức bài toán Retrieval Augmented<br>Generation (RAG).|- Hiểu được các nội dung cơ bản về<br>LLMs và RAG. Input Output của bài<br>toán RAG và luồng xửlý cơ bản.|
|II.|- Các kiến thức cơ bản vềthư viện<br>LangChain.<br>- Tổng quan các cách bước cơ bản trong<br>việc sửdụng LangChain.<br>- Chức năng một sốhàm cơ bản trong<br>LangChain nhằm phục cho cài đặt ứng<br>dụng RAG.<br>- Khái niệm vềAPI và luồng triển khai<br>API cơ bản.|- Nắm được các nội dung và chức năng<br>cơ bản của thư viện LangChain.<br>- Có thểsửdụng thư viện LangChain<br>đểcài đặt một ứng dụng RAG sửdụng<br>LLMs.|



_**- Hết -**_


18
