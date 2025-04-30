from typing_extensions import TypedDict
from langchain_community.utilities import SQLDatabase
import os
from dotenv import load_dotenv
from langchain import hub
from langchain.chat_models import init_chat_model
from typing_extensions import Annotated
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph import graph as langgraph
from langchain_core.messages import HumanMessage, AIMessage


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


def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}


def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}


def chat_session():
    """Run an interactive chat session."""
    print("SQL Chatbot started. Type 'exit' to end the conversation.\n")

    session_id = "chat_session_" + str(os.urandom(4).hex())
    config = {"configurable": {"thread_id": session_id}}

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        try:
            initial_state = {"question": user_input}
            initial_result = graph.invoke(initial_state, config)

            print(f"\nGenerated SQL query: {initial_result['query']}")

            user_approval = input("Do you want to execute this query? (yes/no): ")
            if user_approval.lower() != "yes":
                print("Query execution cancelled.\n")
                continue

            execute_query_tool = QuerySQLDatabaseTool(db=db)
            query_result = execute_query_tool.invoke(initial_result['query'])
            print(f"\nQuery result: {query_result}")

            answer_prompt = (
                "Given the following user question, corresponding SQL query, "
                "and SQL result, answer the user question.\n\n"
                f'Question: {user_input}\n'
                f'SQL Query: {initial_result["query"]}\n'
                f'SQL Result: {query_result}'
            )
            answer = llm.invoke(answer_prompt)
            print(f"\nAnswer: {answer.content}\n")

        except Exception as e:
            print(f"An error occurred: {str(e)}\n")


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

    builder = langgraph.StateGraph(State)
    builder.add_node("write_query", write_query)
    builder.set_entry_point("write_query")
    graph = builder.compile()

    chat_session()