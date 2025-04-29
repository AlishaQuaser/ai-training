from typing_extensions import TypedDict
from langchain_community.utilities import SQLDatabase
import os
from dotenv import load_dotenv
from langchain import hub
from langchain.chat_models import init_chat_model
from typing_extensions import Annotated

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

def execute_query(query):
    """Execute the SQL query and return results."""
    try:
        result = db.run(query)
        return result
    except Exception as e:
        return f"Error executing query: {str(e)}"

if __name__ == "__main__":
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key is None:
        print("You need to set your MISTRAL_API_KEY environment variable")
        exit(1)

    llm = init_chat_model("mistral-large-latest", model_provider="mistralai")

    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    print(f"Database dialect: {db.dialect}")
    print(f"Available tables: {db.get_usable_table_names()}")

    query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

    print("\nPrompt Template:")
    assert len(query_prompt_template.messages) == 2
    for message in query_prompt_template.messages:
        message.pretty_print()

    question = "How many Employees are there?"
    print(f"\nQuestion: {question}")

    state = {"question": question, "query": "", "result": "", "answer": ""}
    query_result = write_query(state)

    state["query"] = query_result["query"]
    print(f"Generated query: {state['query']}")

    state["result"] = execute_query(state["query"])
    print(f"Query result: {state['result']}")