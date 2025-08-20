import os
from qdrant_client import QdrantClient, models

qdrant_client = QdrantClient(
    url=os.environ.get('QDRANT_URL'),
    api_key=os.environ.get('QDRANT_API_KEY'),
)
print(qdrant_client.get_collections())

# qdrant_client.recreate_collection(
#     collection_name="programming",
#     vectors_config={
#         "my_vector_name": models.VectorParams(size=1536, distance=models.Distance.COSINE),
#     },
# )
print()
print(os.environ.get('HF_API_KEY'))
print(os.environ.get('TOGETHER_API_KEY'))
print(os.environ.get('QDRANT_URL'))
print(os.environ.get('QDRANT_API_KEY'))
print(os.environ.get('CEREBRAS_API_KEY'))

"""
Debugging FastAPI:
uvicorn app.py:app --reload

MacOS:
export TOGETHER_API_KEY="YOUR_API_KEY"

Windows:
$env:CEREBRAS_API_KEY = "your_key"
$env:QDRANT_URL = "your_url"
$env:QDRANT_API_KEY = "your_key"
"""
