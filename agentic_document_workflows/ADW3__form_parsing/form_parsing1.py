import os
import json
import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.llms.mistralai import MistralAI

nest_asyncio.apply()
load_dotenv()

def extract_form_fields(file_path):
    llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    mistral_api_key = os.getenv("MISTRAL_API_KEY")

    parser = LlamaParse(
        api_key=llama_cloud_api_key,
        base_url=os.getenv("LLAMA_CLOUD_BASE_URL"),
        result_type="markdown",
        content_guideline_instruction="This is a job application form. Create a list of all the fields that need to be filled in.",
        formatting_instruction="Return a bulleted list of the fields ONLY."
    )

    result = parser.load_data(file_path)[0]
    print(result.text)

    llm = MistralAI(model="mistral-medium", api_key=mistral_api_key)

    raw_json = llm.complete(
        prompt=f"""
        This is a parsed form.
        Convert it into a JSON object containing only the list 
        of fields to be filled in, in the form {{ "fields": [...] }}. 
        <form>{result.text}</form>. 
        Return JSON ONLY, no markdown.
        """
    )

    fields = json.loads(raw_json.text)["fields"]
    return fields


if __name__ == "__main__":
    form_path = "C:/Users/user/Desktop/fake_resume.pdf"
    fields = extract_form_fields(form_path)

    print("\nExtracted Fields:")
    for field in fields:
        print(f"- {field}")
