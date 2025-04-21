import os, json, re, hashlib, random, time
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.mistralai import MistralAI
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
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

load_dotenv()
nest_asyncio.apply()

PERSIST_DIR = "./resume_index_storage"
CACHE_DIR = "./query_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5)
)
def api_call_with_retry(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if "rate limit" in str(e).lower():
            time.sleep(random.uniform(1, 3))
            raise
        else:
            raise


def get_cache_key(text):
    return hashlib.md5(text.encode()).hexdigest()


def get_cached_response(text):
    cache_key = get_cache_key(text)
    cache_file = f"{CACHE_DIR}/{cache_key}.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    return None


def cache_response(text, response):
    cache_key = get_cache_key(text)
    cache_file = f"{CACHE_DIR}/{cache_key}.json"
    with open(cache_file, "w") as f:
        json.dump(response, f)


def setup_mistral_llm():
    load_dotenv()
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    return MistralAI(model="mistral-large-latest", api_key=mistral_api_key)

def setup_mistral_embedding():
    load_dotenv()
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    return MistralAIEmbedding(model="mistral-embed", api_key=mistral_api_key)


Settings.embed_model = setup_mistral_embedding()


def extract_json(text):
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    return json.loads(text)


def chunk_text(text, chunk_size=100):
    """Chunk the text into smaller parts to avoid size limitations."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


class ParseFormEvent(Event):
    application_form: str


class QueryEvent(Event):
    query: str
    field: str


class ResponseEvent(Event):
    response: str
    field: str


class ProcessNextFieldEvent(Event):
    pass


class RAGWorkflow(Workflow):
    storage_dir = "./storage"
    llm: MistralAI
    query_engine: VectorStoreIndex

    @step
    async def set_up(self, ctx: Context, ev: StartEvent) -> ParseFormEvent:
        if not ev.resume_file:
            raise ValueError("No resume file provided")

        if not ev.application_form:
            raise ValueError("No application form provided")

        self.llm = MistralAI(model="mistral-large-latest")

        if os.path.exists(self.storage_dir):
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            index = load_index_from_storage(storage_context)
        else:
            parser = LlamaParse(
                api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                base_url=os.getenv("LLAMA_CLOUD_BASE_URL"),
                result_type="markdown",
                content_guideline_instruction="This is a resume, gather related facts together and format it as bullet points with headers"
            )

            await asyncio.sleep(1)
            documents = api_call_with_retry(parser.load_data, ev.resume_file)

            chunked_documents = []
            for doc in documents:
                chunked_documents.extend(chunk_text(doc.text, chunk_size=500))

            index = VectorStoreIndex.from_documents(
                [{"text": chunk} for chunk in chunked_documents],
                embed_model=MistralAIEmbedding(model_name="mistral-embed")
            )
            index.storage_context.persist(persist_dir=self.storage_dir)

        self.query_engine = index.as_query_engine(llm=self.llm, similarity_top_k=5)

        return ParseFormEvent(application_form=ev.application_form)

    @step
    async def parse_form(self, ctx: Context, ev: ParseFormEvent) -> ProcessNextFieldEvent:
        cache_key = f"form_{get_cache_key(ev.application_form)}"
        cached_fields = get_cached_response(cache_key)

        if cached_fields:
            fields = cached_fields
        else:
            parser = LlamaParse(
                api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                base_url=os.getenv("LLAMA_CLOUD_BASE_URL"),
                result_type="markdown",
                content_guideline_instruction="This is a job application form. Create a list of all the fields that need to be filled in.",
                formatting_instruction="Return a bulleted list of the fields ONLY."
            )

            await asyncio.sleep(2)
            result = api_call_with_retry(parser.load_data, ev.application_form)[0]

            await asyncio.sleep(1)

            raw_json = api_call_with_retry(
                self.llm.complete,
                f"""
                This is a parsed form. 
                Convert it into a JSON object containing only the list 
                of fields to be filled in, in the form {{ fields: [...] }}. 
                <form>{result.text}</form>. 
                Return JSON ONLY, no markdown.
                """
            )

            fields = extract_json(raw_json.text)["fields"]
            cache_response(cache_key, fields)

        await ctx.set("fields", fields)
        await ctx.set("responses", [])
        await ctx.set("current_index", 0)

        return ProcessNextFieldEvent()

    @step
    async def process_next_field(self, ctx: Context, ev: ProcessNextFieldEvent) -> QueryEvent | StopEvent:
        fields = await ctx.get("fields")
        current_index = await ctx.get("current_index")

        if current_index >= len(fields):
            return await self.compile_results(ctx)

        current_field = fields[current_index]
        await ctx.set("current_index", current_index + 1)

        return QueryEvent(
            field=current_field,
            query=f"How would you answer this question about the candidate? {current_field}"
        )

    @step
    async def ask_question(self, ctx: Context, ev: QueryEvent) -> ProcessNextFieldEvent:
        cache_key = f"query_{get_cache_key(ev.query)}"
        cached_response = get_cached_response(cache_key)

        if cached_response:
            response_text = cached_response
        else:
            await asyncio.sleep(2)

            response = api_call_with_retry(
                self.query_engine.query,
                f"This is a question about the specific resume we have in our database: {ev.query}"
            )
            response_text = response.response

            cache_response(cache_key, response_text)

        responses = await ctx.get("responses")
        responses.append({"field": ev.field, "response": response_text})
        await ctx.set("responses", responses)

        return ProcessNextFieldEvent()

    async def compile_results(self, ctx: Context) -> StopEvent:
        responses = await ctx.get("responses")

        responseList = "\n".join(f"Field: {r['field']}\nResponse: {r['response']}" for r in responses)

        await asyncio.sleep(2)

        result = api_call_with_retry(
            self.llm.complete,
            f"""
            You are given a list of fields in an application form and responses to
            questions about those fields from a resume. Combine the two into a list of
            fields and succinct, factual answers to fill in those fields.

            <responses>
            {responseList}
            </responses>
            """
        )

        return StopEvent(result=result.text)


async def main():
    w = RAGWorkflow(timeout=300, verbose=False)
    result = await w.run(
        resume_file="C:/Users/user/Desktop/fake_resume.pdf",
        application_form="C:/Users/user/Desktop/application_form.pdf"
    )
    print(result)

    os.makedirs("workflows", exist_ok=True)
    WORKFLOW_FILE = "workflows/form_parsing_generate_ques.html"
    draw_all_possible_flows(w, filename=WORKFLOW_FILE)

if __name__ == "__main__":
    asyncio.run(main())
