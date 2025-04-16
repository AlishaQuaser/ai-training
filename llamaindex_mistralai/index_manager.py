import os
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llamaindex_mistralai.base import setup_mistral, load_documents

PERSIST_DIR = "./llamaindex_mistralai/index_storage"


def create_index_from_documents(documents):
    print("Creating index with Mistral embeddings...")
    setup_mistral()
    document_index = VectorStoreIndex(documents)
    document_index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("Index created and saved successfully")
    return document_index


def load_or_create_index(documents_path):
    if os.path.exists(PERSIST_DIR):
        print("Loading index from disk...")
        setup_mistral()
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        print("Index loaded successfully")
    else:
        print("Creating new index...")
        documents = load_documents(documents_path)
        index = create_index_from_documents(documents)
    return index
