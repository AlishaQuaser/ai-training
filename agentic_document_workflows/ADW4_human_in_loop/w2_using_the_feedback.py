import os, json, re, hashlib, random, time
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Document
)
from llama_index.core.settings import Settings
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

PERSIST_DIR = "./storage2"
CACHE_DIR = "./query_cache2"
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


def chunk_text(text, chunk_size=500):
    """Chunk the text into smaller parts to avoid size limitations."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


Settings.embed_model = setup_mistral_embedding()


def extract_json(text):
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            print("Warning: Could not parse JSON from code block")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("Warning: Could not parse JSON, returning empty object")
        print(f"JSON text was: {text[:100]}...")
        return {"fields": []}


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
    is_feedback_iteration: bool = False


class RAGWorkflow(Workflow):
    storage_dir = "./storage2"
    llm: MistralAI
    query_engine: VectorStoreIndex

    @step
    async def set_up(self, ctx: Context, ev: StartEvent) -> ParseFormEvent:
        await ctx.set("responses", {})

        if not ev.resume_file:
            raise ValueError("No resume file provided")

        if not ev.application_form:
            raise ValueError("No application form provided")

        self.llm = setup_mistral_llm()

        if os.path.exists(self.storage_dir):
            print("Loading existing index from storage")
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            index = load_index_from_storage(storage_context)
        else:
            print("Creating new index from resume")
            cache_key = f"resume_{get_cache_key(ev.resume_file)}"
            cached_documents = get_cached_response(cache_key)

            if cached_documents:
                print("Using cached resume content")
                documents_text = cached_documents
                documents = [Document(text=doc_text) for doc_text in documents_text]
            else:
                print("Parsing resume with LlamaParse")
                parser = LlamaParse(
                    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                    base_url=os.getenv("LLAMA_CLOUD_BASE_URL"),
                    result_type="markdown",
                    content_guideline_instruction="This is a resume, gather related facts together and format it as bullet points with headers"
                )
                await asyncio.sleep(1)
                documents = api_call_with_retry(parser.load_data, ev.resume_file)

                documents_text = [doc.text for doc in documents]
                cache_response(cache_key, documents_text)

            chunked_documents = []
            for doc in documents:
                chunks = chunk_text(doc.text, chunk_size=500)
                chunked_documents.extend([Document(text=chunk) for chunk in chunks])

            print(f"Created {len(chunked_documents)} chunks from resume")

            index = VectorStoreIndex.from_documents(
                chunked_documents,
                embed_model=setup_mistral_embedding()
            )
            index.storage_context.persist(persist_dir=self.storage_dir)
            print("Index created and saved to storage")

        self.query_engine = index.as_query_engine(llm=self.llm, similarity_top_k=8)

        return ParseFormEvent(application_form=ev.application_form)

    @step
    async def parse_form(self, ctx: Context, ev: ParseFormEvent) -> GenerateQuestionsEvent:
        cache_key = f"form_{get_cache_key(ev.application_form)}"
        cached_fields = get_cached_response(cache_key)

        if cached_fields:
            print("Using cached form fields")
            fields = cached_fields
        else:
            print("Parsing application form with LlamaParse")
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
            print("Extracting fields from parsed form")
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

            json_data = extract_json(raw_json.text)
            fields = json_data.get("fields", [])

            if not fields:
                print("WARNING: No fields extracted from form. Raw JSON response:")
                print(raw_json.text[:500])
                fields = ["Field extraction failed - please check the application form"]
            else:
                print(f"Extracted {len(fields)} fields from form")

            cache_response(cache_key, fields)

        await ctx.set("fields_to_fill", fields)
        await ctx.set("responses", {})
        return GenerateQuestionsEvent(is_feedback_iteration=False)

    @step
    async def generate_questions(self, ctx: Context, ev: GenerateQuestionsEvent | FeedbackEvent) -> QueryEvent:
        fields = await ctx.get("fields_to_fill")
        print(f"Generating questions for {len(fields)} fields")

        is_feedback_iteration = False
        feedback = ""

        if isinstance(ev, FeedbackEvent):
            feedback = ev.feedback
            print(f"Processing with feedback: {feedback[:100]}...")
            is_feedback_iteration = True
        elif hasattr(ev, "is_feedback_iteration"):
            is_feedback_iteration = ev.is_feedback_iteration

        await ctx.set("is_feedback_iteration", is_feedback_iteration)
        await ctx.set("feedback", feedback)

        for field in fields:
            question = f"How would you answer this question about the candidate? <field>{field}</field>"

            if feedback:
                question += f"""
                    \nWe previously got feedback about how we answered the questions.
                    It might not be relevant to this particular field, but here it is:
                    <feedback>{feedback}</feedback>
                """

            ctx.send_event(QueryEvent(
                field=field,
                query=question
            ))

        await ctx.set("total_fields", len(fields))
        return

    @step
    async def ask_question(self, ctx: Context, ev: QueryEvent) -> ResponseEvent:
        print(f"Processing field: {ev.field}")

        try:
            responses = await ctx.get("responses")
        except:
            responses = {}
            await ctx.set("responses", responses)

        is_feedback_iteration = False
        try:
            is_feedback_iteration = await ctx.get("is_feedback_iteration")
        except:
            await ctx.set("is_feedback_iteration", False)

        if not is_feedback_iteration:
            cache_key = f"query_{get_cache_key(ev.query)}"
            cached_response = get_cached_response(cache_key)

            if cached_response:
                print(f"Using cached response for field: {ev.field}")

                responses[ev.field] = cached_response
                await ctx.set("responses", responses)

                return ResponseEvent(field=ev.field, response=cached_response)

        await asyncio.sleep(1)
        print(f"Querying resume data for field: {ev.field}")

        enhanced_query = f"This is a question about the specific resume we have in our database: {ev.query}"

        response = api_call_with_retry(
            self.query_engine.query,
            enhanced_query
        )
        response_text = response.response

        print(f"Response for {ev.field}: {response_text[:100]}...")

        if not is_feedback_iteration:
            cache_response(ev.query, response_text)

        responses[ev.field] = response_text
        await ctx.set("responses", responses)

        return ResponseEvent(field=ev.field, response=response_text)

    @step
    async def fill_in_application(self, ctx: Context, ev: ResponseEvent) -> InputRequiredEvent:
        total_fields = await ctx.get("total_fields")

        responses = ctx.collect_events(ev, [ResponseEvent] * total_fields)
        if responses is None:
            return None

        print(f"Collected all {len(responses)} responses, generating final form")

        try:
            stored_responses = await ctx.get("responses")
        except:
            stored_responses = {r.field: r.response for r in responses}
            await ctx.set("responses", stored_responses)

        responseList = "\n".join(
            f"Field: {r.field}\nResponse: {stored_responses.get(r.field, r.response)}"
            for r in responses
        )

        feedback = ""
        try:
            feedback = await ctx.get("feedback") or ""
        except:
            pass

        feedback_prompt = ""
        if feedback:
            feedback_prompt = f"""
            Consider this feedback when preparing the final answers:
            <feedback>
            {feedback}
            </feedback>
            """

        await asyncio.sleep(1)
        result = api_call_with_retry(
            self.llm.complete,
            f"""
            You are given a list of fields in an application form and responses to
            questions about those fields from a resume. Combine the two into a list of
            fields and specific, detailed answers to fill in those fields. Do not use
            "not provided" unless there's absolutely no relevant information in the responses.
            Be creative but factual based on what's in the responses.
            
            {feedback_prompt}

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

        if "OKAY" in verdict:
            verdict = "OKAY"
        elif "FEEDBACK" in verdict:
            verdict = "FEEDBACK"
        else:
            verdict = "FEEDBACK"

        print(f"LLM says the verdict was {verdict}")
        if verdict == "OKAY":
            return StopEvent(result=await ctx.get("filled_form"))
        else:
            return FeedbackEvent(feedback=ev.response)


async def main():
    import shutil
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    w = RAGWorkflow(timeout=600, verbose=True)
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

    # os.makedirs("workflows", exist_ok=True)
    # WORKFLOW_FILE = "workflows/form_filling_with_feedback_impl.html"
    # draw_all_possible_flows(w, filename=WORKFLOW_FILE)


if __name__ == "__main__":
    asyncio.run(main())