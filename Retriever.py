import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
from typing import List, Optional
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, SentenceTransformer
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

# Function to split documents into chunks
def split_documents(chunk_size: int, knowledge_base: List[LangchainDocument], tokenizer_name: str) -> List[LangchainDocument]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=[
            "\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""
        ],
    )
    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])
    return docs_processed

# Load the knowledge base
def load_knowledge_base():
    ds = load_dataset("MedRAG", "textbooks", split="train")
    return [
        LangchainDocument(page_content=doc["text"], metadata={"id": doc["id"]})
        for doc in tqdm(ds, desc="Loading Knowledge Base")
    ]

# Main function to process documents and create the vector database
def create_vector_database(embedding_model_name: str, knowledge_base: List[LangchainDocument]):
    docs_processed = split_documents(
        chunk_size=256,  # Adjust chunk size as needed
        knowledge_base=knowledge_base,
        tokenizer_name=embedding_model_name,
    )
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        multi_process=True,
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_database = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    return vector_database

# Example usage
if __name__ == "__main__":
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    knowledge_base = load_knowledge_base()
    vector_database = create_vector_database(EMBEDDING_MODEL_NAME, knowledge_base)
    print("Vector database created successfully.")
