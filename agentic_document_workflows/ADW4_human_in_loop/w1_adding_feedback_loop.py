import os, json, re, hashlib, random, time
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.llms.mistralai import MistralAI
from llama_index.core.settings import Settings
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
    InputRequiredEvent,
    HumanResponseEvent
)
from llama_index.utils.workflow import draw_all_possible_flows
import nest_asyncio
import asyncio
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

load_dotenv()
nest_asyncio.apply()

PERSIST_DIR = "./storage"
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
    try:
        return json.loads(text)
    except:
        print("Warning: Could not parse JSON, returning empty object")
        return {}


class ParseFormEvent(Event):
    application_form: str


class QueryEvent(Event):
    query: str
    field: str


class ResponseEvent(Event):
    response: str
    field: str


class FeedbackEvent(Event):
    feedback: str


class GenerateQuestionsEvent(Event):
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

        self.llm = setup_mistral_llm()

        if os.path.exists(self.storage_dir):
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            index = load_index_from_storage(storage_context)
        else:
            cache_key = f"resume_{get_cache_key(ev.resume_file)}"
            cached_documents = get_cached_response(cache_key)

            if cached_documents:
                documents = cached_documents
            else:
                parser = LlamaParse(
                    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                    base_url=os.getenv("LLAMA_CLOUD_BASE_URL"),
                    result_type="markdown",
                    content_guideline_instruction="This is a resume, gather related facts together and format it as bullet points with headers"
                )
                await asyncio.sleep(1)
                documents = api_call_with_retry(parser.load_data, ev.resume_file)

                cache_response(cache_key, [doc.text for doc in documents])

            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=setup_mistral_embedding()
            )
            index.storage_context.persist(persist_dir=self.storage_dir)

        self.query_engine = index.as_query_engine(llm=self.llm, similarity_top_k=5)

        return ParseFormEvent(application_form=ev.application_form)

    @step
    async def parse_form(self, ctx: Context, ev: ParseFormEvent) -> GenerateQuestionsEvent:
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

            await asyncio.sleep(1)
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

            fields = extract_json(raw_json.text).get("fields", [])
            cache_response(cache_key, fields)

        await ctx.set("fields_to_fill", fields)
        return GenerateQuestionsEvent()

    @step
    async def generate_questions(self, ctx: Context, ev: GenerateQuestionsEvent | FeedbackEvent) -> QueryEvent:
        fields = await ctx.get("fields_to_fill")

        for field in fields:
            question = f"How would you answer this question about the candidate? <field>{field}</field>"
            ctx.send_event(QueryEvent(
                field=field,
                query=question
            ))

        await ctx.set("total_fields", len(fields))
        return

    @step
    async def ask_question(self, ctx: Context, ev: QueryEvent) -> ResponseEvent:
        cache_key = f"query_{get_cache_key(ev.query)}"
        cached_response = get_cached_response(cache_key)

        if cached_response:
            response_text = cached_response
        else:
            await asyncio.sleep(1)
            response = api_call_with_retry(
                self.query_engine.query,
                f"This is a question about the specific resume we have in our database: {ev.query}"
            )
            response_text = response.response
            cache_response(cache_key, response_text)

        return ResponseEvent(field=ev.field, response=response_text)

    @step
    async def fill_in_application(self, ctx: Context, ev: ResponseEvent) -> InputRequiredEvent:
        total_fields = await ctx.get("total_fields")

        responses = ctx.collect_events(ev, [ResponseEvent] * total_fields)
        if responses is None:
            return None

        responseList = "\n".join(f"Field: {r.field}\nResponse: {r.response}" for r in responses)

        await asyncio.sleep(1)
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

        await ctx.set("filled_form", str(result))

        return InputRequiredEvent(
            prefix="How does this look? Give me any feedback you have on any of the answers.",
            result=result
        )

    @step
    async def get_feedback(self, ctx: Context, ev: HumanResponseEvent) -> FeedbackEvent | StopEvent:
        await asyncio.sleep(1)
        result = api_call_with_retry(
            self.llm.complete,
            f"""
            You have received some human feedback on the form-filling task you've done.
            Does everything look good, or is there more work to be done?
            <feedback>
            {ev.response}
            </feedback>
            If everything is fine, respond with just the word 'OKAY'.
            If there's any other feedback, respond with just the word 'FEEDBACK'.
            """
        )

        verdict = result.text.strip()

        print(f"LLM says the verdict was {verdict}")
        if verdict == "OKAY":
            return StopEvent(result=await ctx.get("filled_form"))
        else:
            return FeedbackEvent(feedback=ev.response)


async def main():
    w = RAGWorkflow(timeout=600, verbose=False)
    handler = w.run(
        resume_file="C:/Users/user/Desktop/fake_resume.pdf",
        application_form="C:/Users/user/Desktop/application_form.pdf"
    )

    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            print("We've filled in your form! Here are the results:\n")
            print(event.result)
            response = input(event.prefix + " ")
            handler.ctx.send_event(
                HumanResponseEvent(
                    response=response
                )
            )

    response = await handler
    print("Agent complete! Here's your final result:")
    print(str(response))

    os.makedirs("workflows", exist_ok=True)
    WORKFLOW_FILE = "workflows/form_filling_with_feedback.html"
    draw_all_possible_flows(w, filename=WORKFLOW_FILE)


if __name__ == "__main__":
    asyncio.run(main())