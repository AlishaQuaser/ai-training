import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SimpleNodeParser

load_dotenv()


def load_documents(directory_path):
    print(f"Loading and splitting documents from {directory_path}...")
    documents = SimpleDirectoryReader(directory_path).load_data()
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    print(f"Split into {len(nodes)} chunks")
    return nodes


def setup_mistral():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set in .env file")

    llm = MistralAI(model="mistral-large-latest", api_key=api_key)
    embed_model = MistralAIEmbedding(model_name="mistral-embed", api_key=api_key)

    Settings.llm = llm
    Settings.embed_model = embed_model
