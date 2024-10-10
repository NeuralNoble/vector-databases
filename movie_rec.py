import certifi
import pymongo
import ssl
from dotenv import load_dotenv
import os
import requests

load_dotenv()

db_uri = os.getenv("DB_URI")
hf_token = os.getenv("HF_TOKEN")
embedding_url = os.getenv("EMBEDDING_URL")

client = pymongo.MongoClient(
    db_uri,
    tlsCAFile=certifi.where())
db = client.sample_mflix
movies = db.movies


def generate_embedding(text: str) -> list:
    response = requests.post(embedding_url, headers={"Authorization": f"Bearer {hf_token}"}, json={"inputs": text})

    if response.status_code != 200:
        raise ValueError(f"request failed with status code {response.status_code}: {response.text}")

    return response.json()


query = "imaginary characters from outer space at war"

# for doc in movies.find({'plot':{"$exists": True}}).limit(50):
#   doc['i'] = generate_embedding(doc['plot'])
#   movies.replace_one({'_id': doc['_id']}, doc)

# print(generate_embedding("aman"))

result = movies.aggregate([
    {"$vectorSearch": {
        "queryVector": generate_embedding(query),
        "path": "plot_embedding_hf",
        "numCandidates": 100,
        "limit": 4,
        "index": "PlotSemanticSearch",
    }}
])

for doc in result:
    print(f"Movie Name: {doc['title']},\nMovie Plot: {doc['plot']}\n")
