import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.mistralai import MistralAI  #
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core.settings import Settings
from llama_index.core import VectorStoreIndex
from llama_parse import LlamaParse
from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context
)
import nest_asyncio
import asyncio

load_dotenv()
nest_asyncio.apply()

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

class QueryEvent(Event):
    query: str

class RAGWorkflow(Workflow):
    storage_dir = "./storage"
    llm: MistralAI
    query_engine: VectorStoreIndex

    @step
    async def set_up(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        if not ev.resume_file:
            raise ValueError("No resume file provided")

        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")

        self.llm = MistralAI(model="mistral-large-latest", api_key=mistral_api_key)
        embed_model = MistralAIEmbedding(model="mistral-embed", api_key=mistral_api_key)

        Settings.embed_model = embed_model

        if os.path.exists(self.storage_dir):
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            index = load_index_from_storage(storage_context)
        else:
            documents = LlamaParse(
                result_type="markdown",
                system_prompt="This is a resume, gather related facts together and format it as bullet points with headers"
            ).load_data(ev.resume_file)

            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=embed_model
            )
            index.storage_context.persist(persist_dir=self.storage_dir)

        self.query_engine = index.as_query_engine(
            llm=self.llm,
            similarity_top_k=5,
            embed_model=embed_model
        )

        return QueryEvent(query=ev.query)

    @step
    async def ask_question(self, ctx: Context, ev: QueryEvent) -> StopEvent:
        response = self.query_engine.query(f"This is a question about the specific resume we have in our database: {ev.query}")
        return StopEvent(result=response.response)
async def main():
    w = RAGWorkflow(timeout=120, verbose=False)
    result = await w.run(
        resume_file="C:/Users/user/Desktop/fake_resume.pdf",
        query="Where is the first place the applicant worked?"
    )
    print(result)

    #workflow visualization
    WORKFLOW_FILE = "workflows/rag_workflow.html"
    draw_all_possible_flows(w, filename=WORKFLOW_FILE)

if __name__ == "__main__":
    asyncio.run(main())