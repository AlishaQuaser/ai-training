import os
import json
import re
import hashlib
import time
import random
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_parse import LlamaParse
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

CACHE_DIR = "./query_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def setup_mistral_embedding():
    return MistralAIEmbedding(model_name="mistral-embed")

Settings.embed_model = setup_mistral_embedding()

def get_cache_key(text):
    return hashlib.md5(text.encode()).hexdigest()

def get_cached_response(text):
    cache_key = get_cache_key(text)
    cache_file = f"{CACHE_DIR}/{cache_key}.json"
    if os.path.exists(cache_file):
        print(f"Cache hit for: {text[:30]}...")
        with open(cache_file, "r") as f:
            return json.load(f)
    print(f"Cache miss for: {text[:30]}...")
    return None

def cache_response(text, response):
    cache_key = get_cache_key(text)
    cache_file = f"{CACHE_DIR}/{cache_key}.json"
    with open(cache_file, "w") as f:
        json.dump(response, f)
    print(f"Cached response for: {text[:30]}...")

async def api_call_with_backoff(func, *args, **kwargs):
    max_retries = 5
    base_delay = 2

    for attempt in range(max_retries):
        try:
            await asyncio.sleep(1 + random.random())
            return func(*args, **kwargs)
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited. Retrying in {delay:.2f} seconds (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(delay)
                if attempt == max_retries - 1:
                    print("Maximum retries reached. Raising error.")
                    raise
            else:
                print(f"Error not related to rate limiting: {e}")
                raise

class ParseFormEvent(Event):
    application_form: str

class QueryEvent(Event):
    query: str
    field: str

class ResponseEvent(Event):
    response: str
    field: str

def extract_json(text):
    """Extract JSON from text that might contain markdown or other content."""
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_match = re.search(r'({[\s\S]*})', text)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = text.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        field_matches = re.findall(r'"([^"]+)"', json_str)
        if field_matches:
            return {"fields": field_matches}

        lines = text.strip().split('\n')
        fields = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('```', '{', '}')):
                line = re.sub(r'^[\s\-\*\d\.\)\>"\']+', '', line).strip()
                if line:
                    fields.append(line)

        return {"fields": fields}

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

            await asyncio.sleep(2)
            documents = await api_call_with_backoff(parser.load_data, ev.resume_file)

            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=MistralAIEmbedding(model_name="mistral-embed")
            )
            index.storage_context.persist(persist_dir=self.storage_dir)

        self.query_engine = index.as_query_engine(llm=self.llm, similarity_top_k=5)

        return ParseFormEvent(application_form=ev.application_form)

    @step
    async def parse_form(self, ctx: Context, ev: ParseFormEvent) -> QueryEvent:
        cache_key = f"form_{get_cache_key(ev.application_form)}"
        cached_fields = get_cached_response(cache_key)

        if cached_fields:
            fields = cached_fields
            print(f"Using cached fields: {fields[:3]}...")

        else:
            parser = LlamaParse(
                api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                base_url=os.getenv("LLAMA_CLOUD_BASE_URL"),
                result_type="markdown",
                content_guideline_instruction="This is a job application form. Create a list of all the fields that need to be filled in.",
                formatting_instruction="Return a bulleted list of the fields ONLY."
            )

            await asyncio.sleep(2)
            result = await api_call_with_backoff(parser.load_data, ev.application_form)
            result = result[0]

            await asyncio.sleep(2)

            raw_json = await api_call_with_backoff(
                self.llm.complete,
                f"""
            This is a parsed form. 
            Convert it into a JSON object containing only the list 
            of fields to be filled in, in the form {{ "fields": [...] }}. 
            <form>{result.text}</form>. 
            Return JSON ONLY, no markdown or explanation.
            Format as: {{ "fields": ["field1", "field2", ...] }}
            """
            )

            print("Raw LLM response:", raw_json.text)

            json_data = extract_json(raw_json.text)

            if "fields" not in json_data:
                if isinstance(json_data, list):
                    fields = json_data
                else:
                    fields = list(json_data.values())
                    if len(fields) == 0 or not isinstance(fields[0], list):
                        fields = [str(v) for v in json_data.values()]
            else:
                fields = json_data["fields"]

            cache_response(cache_key, fields)

            print(f"Extracted {len(fields)} fields")

            await ctx.set("total_fields", len(fields))

            for i in range(1, len(fields)):
                await asyncio.sleep(0.5)
                ctx.send_event(QueryEvent(
                    field=fields[i],
                    query=f"How would you answer this question about the candidate? {fields[i]}"
                ))

            if fields:
                return QueryEvent(
                    field=fields[0],
                    query=f"How would you answer this question about the candidate? {fields[0]}"
                )
            else:
                return StopEvent(result="No fields found in the application form.")

        @step
        async def ask_question(self, ctx: Context, ev: QueryEvent) -> ResponseEvent:
            total_fields = await ctx.get("total_fields", None)
            if total_fields is not None:
                await ctx.set("total_fields", total_fields)

            cache_key = f"query_{get_cache_key(ev.query)}"
            cached_response = get_cached_response(cache_key)

            if cached_response:
                response_text = cached_response
            else:
                await asyncio.sleep(2)

                query_text = f"This is a question about the specific resume we have in our database: {ev.query}"
                response = await api_call_with_backoff(self.query_engine.query, query_text)
                response_text = response.response

                cache_response(cache_key, response_text)

            return ResponseEvent(field=ev.field, response=response_text)

        @step
        async def fill_in_application(self, ctx: Context, ev: ResponseEvent) -> StopEvent:
            try:
                total_fields = await ctx.get("total_fields")
            except KeyError:
                all_responses = ctx.collect_all_events([ResponseEvent])
                if all_responses:
                    total_fields = len(all_responses)
                    print(f"Estimated total_fields as {total_fields} from collected events")
                else:
                    total_fields = 1
                    print("Warning: Could not determine total fields, using default of 1")

            print(f"Waiting for {total_fields} ResponseEvents")

            responses = ctx.collect_events(ev, [ResponseEvent] * total_fields)
            if responses is None:
                print("Still waiting for more ResponseEvents...")
                return None

            print(f"Collected all {len(responses)} responses, generating final result")

            responseList = "\n".join("Field: " + r.field + "\n" + "Response: " + r.response for r in responses)

            cache_key = f"final_{get_cache_key(responseList[:200])}"
            cached_final = get_cached_response(cache_key)

            if cached_final:
                print("Using cached final result")
                result_text = cached_final
            else:
                print("Generating new final result")
                await asyncio.sleep(3)

                prompt = f"""
                    You are given a list of fields in an application form and responses to
                    questions about those fields from a resume. Combine the two into a list of
                    fields and succinct, factual answers to fill in those fields.
        
                    <responses>
                    {responseList}
                    </responses>
                """

                result = await api_call_with_backoff(self.llm.complete, prompt)
                result_text = result.text

                cache_response(cache_key, result_text)

            return StopEvent(result=result_text)

if __name__ == "__main__":
    import asyncio
    from llama_index.utils.workflow import draw_all_possible_flows

    async def main():

        w = RAGWorkflow(timeout=600, verbose=True)
        result = await w.run(
            resume_file="C:/Users/user/Desktop/fake_resume.pdf",
            application_form="C:/Users/user/Desktop/fake_application_form.pdf"
        )
        print(result)

        os.makedirs("workflows", exist_ok=True)
        WORKFLOW_FILE = "workflows/form_parsing_workflow.html"
        draw_all_possible_flows(w, filename=WORKFLOW_FILE)

    asyncio.run(main())