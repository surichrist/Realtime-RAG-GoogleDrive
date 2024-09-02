import streamlit as st
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.llms.groq import Groq
from redisvl.schema import IndexSchema
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
os.environ["GROQ_API_KEY"] = ""
llm = Groq(model="llama3-70b-8192", api_key="")
Settings.llm = Groq(model="llama3-70b-8192", api_key="")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
custom_schema = IndexSchema.from_dict(
    {
        "index": {"name": "gdrive", "prefix": "doc"},
        # customize fields that are indexed
        "fields": [
            # required fields for llamaindex
            {"type": "tag", "name": "id"},
            {"type": "tag", "name": "doc_id"},
            {"type": "text", "name": "text"},
            # custom vector field for bge-small-en-v1.5 embeddings
            {
                "type": "vector",
                "name": "vector",
                "attrs": {
                    "dims": 384,
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                },
            },
        ],
    }
)

vector_store = RedisVectorStore(
    schema=custom_schema,
    redis_url="redis://localhost:6379",
)

# Set up the ingestion cache layer
cache = IngestionCache(
    cache=RedisCache.from_host_and_port("localhost", 6379),
    collection="redis_cache",
)
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        embed_model,
    ],
    docstore=RedisDocumentStore.from_host_and_port(
        "localhost", 6379, namespace="document_store"
    ),
    vector_store=vector_store,
    cache=cache,
    docstore_strategy=DocstoreStrategy.UPSERTS,
)

from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_vector_store(
    pipeline.vector_store, embed_model=embed_model
)

from llama_index.readers.google import GoogleDriveReader

loader = GoogleDriveReader(service_account_key_path="/Users/raghavsuri/Downloads/llm2348543-87027f51bc69.json")


def load_data(folder_id: str):
    docs = loader.load_data(folder_id=folder_id)
    for doc in docs:
        doc.id_ = doc.metadata["file name"]
    return docs



from llama_index.llms.groq import Groq



# Title of the application
st.title('Real Time RAG from Google Drive')



# User input section
user_input = st.text_input("You:", key="input")



# Button to send the input
if st.button("Send"):
    docs = load_data(folder_id="1M-RT_O5b-bF1-Wlk0kwtrdVlav1Ngxs8")
    nodes = pipeline.run(documents=docs)
    query_engine = index.as_query_engine()
    response = query_engine.query(user_input)
    
    st.write(str(response))        







