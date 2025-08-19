import os
from qdrant_client import QdrantClient, models

qdrant_client = QdrantClient(
    url="https://61ad2193-327e-49c2-9a92-a890808fc6f9.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.9E6v7XTFaF8Vq9ZQ1EtZ10ctEorpvCpEC8SUxMHD8VA",
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
# cerebras API: csk-dm4364nx3xkx2d2rffjn6hxy6den2f934n3envexew4d45p4


"""
Debugging FastAPI:
uvicorn app.py:app --reload

MacOS:
export TOGETHER_API_KEY="YOUR_API_KEY"

Windows:
$env:CEREBRAS_API_KEY = "csk-dm4364nx3xkx2d2rffjn6hxy6den2f934n3envexew4d45p4"
$env:QDRANT_URL = "https://61ad2193-327e-49c2-9a92-a890808fc6f9.europe-west3-0.gcp.cloud.qdrant.io"
$env:QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.9E6v7XTFaF8Vq9ZQ1EtZ10ctEorpvCpEC8SUxMHD8VA"
"""


"""Token Count Test
INPUT_token_count:10616
OUTPUT_token_count:4808
AVG_INPUT_token_count:1061.6
AVG_OUTPUT_token_count:480.8
TOTAL_TOKEN_COUNT:[1717 1743 1417 1419 1483 1630 1516 1619 1580 1300]
TOKEN_COUNT_PER_GENERATION - :[15424.]
AVG_TOKEN_COUNT_PER_GENERATION:[np.float64(15424.0), 1]

INPUT_token_count:10299.0
OUTPUT_token_count:5628.0
AVG_INPUT_token_count:1029.9
AVG_OUTPUT_token_count:562.8
TOTAL_TOKEN_COUNT:[1852. 1520. 1615. 1790. 1539. 1562. 1290. 1686. 1460. 1613.]
TOKEN_COUNT_PER_GENERATION - :[15424. 15927.]
AVG_TOKEN_COUNT_PER_GENERATION:[np.float64(15675.5), 2]

INPUT_token_count:9640.0
OUTPUT_token_count:5576.0
AVG_INPUT_token_count:964.0
AVG_OUTPUT_token_count:557.6
TOTAL_TOKEN_COUNT:[1252. 1835. 1490. 1537. 1394. 1620. 1670. 1707. 1458. 1253.]
TOKEN_COUNT_PER_GENERATION - :[15424. 15927. 15216.]
AVG_TOKEN_COUNT_PER_GENERATION:[np.float64(15522.333333333334), 3]

INPUT_token_count:9356.0
OUTPUT_token_count:5277.0
AVG_INPUT_token_count:935.6
AVG_OUTPUT_token_count:527.7
TOTAL_TOKEN_COUNT:[1368. 1295. 1849. 1523. 1468. 1473. 1486. 1426. 1595. 1150.]
TOKEN_COUNT_PER_GENERATION - :[15424. 15927. 15216. 14633.]
AVG_TOKEN_COUNT_PER_GENERATION:[np.float64(15300.0), 4]

INPUT_token_count:9828.0
OUTPUT_token_count:4758.0
AVG_INPUT_token_count:982.8
AVG_OUTPUT_token_count:475.8
TOTAL_TOKEN_COUNT:[1820. 1235. 1911. 1591. 1312. 1242. 1372. 1533. 1393. 1177.]
TOKEN_COUNT_PER_GENERATION - :[15424. 15927. 15216. 14633. 14586.]
AVG_TOKEN_COUNT_PER_GENERATION:[np.float64(15157.2), 5]
"""
