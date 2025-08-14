"""
MacOS:
export TOGETHER_API_KEY="YOUR_API_KEY"

Windows:
$env:TOGETHER_API_KEY = "YOUR_API_KEY"
"""

import os
from qdrant_client import QdrantClient, models

qdrant_client = QdrantClient(
    url="https://61ad2193-327e-49c2-9a92-a890808fc6f9.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.9E6v7XTFaF8Vq9ZQ1EtZ10ctEorpvCpEC8SUxMHD8VA",
)
print(qdrant_client.get_collections())

qdrant_client.recreate_collection(
    collection_name="programming",
    vectors_config={
        "my_vector_name": models.VectorParams(size=1536, distance=models.Distance.COSINE),
    },
)

print(os.environ.get('HF_API_KEY'))
print(os.environ.get('TOGETHER_API_KEY'))
print(os.environ.get('QDRANT_URL'))
print(os.environ.get('QDRANT_API_KEY'))
# print(os.environ.get('GEMINI_KEY')) -> unlimited used ???
