import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
import nest_asyncio
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import load_index_from_storage
from llama_index.llms.mistralai import MistralAI

PERSIST_DIR = "./resume_index_storage"


def setup_mistral():
    """Set up Mistral API embeddings and LLM"""
    load_dotenv()
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    embed_model = MistralAIEmbedding(
        model_name="mistral-embed",
        api_key=mistral_api_key
    )
    llm = MistralAI(
        model="mistral-large-latest",
        api_key=mistral_api_key
    )
    return embed_model, llm


def load_documents(documents_path):
    """Load and parse documents using LlamaParse"""
    load_dotenv()
    nest_asyncio.apply()

    llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

    documents = LlamaParse(
        api_key=llama_cloud_api_key,
        base_url=os.getenv("LLAMA_CLOUD_BASE_URL"),
        result_type="markdown",
        content_guideline_instruction="This is a resume, gather related facts together and format it as bullet points with headers"
    ).load_data(documents_path)

    print(f"Loaded document: {documents[0].text[:100]}...")
    return documents


def create_index_from_documents(documents):
    """Create and persist a new index from documents"""
    embed_model, _ = setup_mistral()

    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model
    )

    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"Index created and saved to {PERSIST_DIR}")

    return index


def load_or_create_index(documents_path):
    """Load index from disk if it exists, otherwise create and save it"""
    if os.path.exists(PERSIST_DIR):
        print("Loading index from disk...")
        embed_model, _ = setup_mistral()
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        print("Index loaded successfully")
    else:
        print("Creating new index...")
        documents = load_documents(documents_path)
        index = create_index_from_documents(documents)
    return index


def create_query_engine(index):
    """Create a query engine using the provided index"""
    _, llm = setup_mistral()

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5
    )

    return query_engine


def query_resume(query_engine, question):
    """Query the resume with a specific question"""
    response = query_engine.query(question)
    print(response)
    return response


if __name__ == "__main__":
    load_dotenv()
    resume_path = "C:/Users/user/Desktop/fake_resume.pdf"

    index = load_or_create_index(resume_path)

    query_engine = create_query_engine(index)
    response = query_resume(query_engine, "What is this person's name and what was their most recent job?")