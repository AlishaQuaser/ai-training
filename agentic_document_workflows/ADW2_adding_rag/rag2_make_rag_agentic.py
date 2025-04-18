import os
from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core.settings import Settings

PERSIST_DIR = "./resume_index_storage"


def setup_mistral_llm():
    load_dotenv()
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    return MistralAI(model="mistral-large-latest", api_key=mistral_api_key)


def setup_mistral_embedding():
    load_dotenv()
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    return MistralAIEmbedding(model="mistral-embed", api_key=mistral_api_key)


def load_query_engine():
    Settings.embed_model = setup_mistral_embedding()

    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

    llm = setup_mistral_llm()

    query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)
    return query_engine, llm


def query_resume(q: str) -> str:
    """Answers questions about a specific resume."""
    response = query_engine.query(
        f"This is a question about the specific resume we have in our database: {q}"
    )
    return response.response


if __name__ == "__main__":
    query_engine, llm = load_query_engine()

    resume_tool = FunctionTool.from_defaults(fn=query_resume)

    agent = FunctionCallingAgent.from_tools(
        tools=[resume_tool],
        llm=llm,
        verbose=True
    )

    response = agent.chat("How many years of experience does the applicant have?")
    print(response)
