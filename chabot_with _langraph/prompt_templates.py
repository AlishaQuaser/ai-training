import getpass
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

load_dotenv()


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


def main():
    if not os.environ.get("MISTRAL_API_KEY"):
        os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")

    model = init_chat_model("mistral-large-latest", model_provider="mistralai")

    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

    workflow = StateGraph(state_schema=State)

    def call_model(state: State):
        prompt = prompt_template.invoke(state)
        response = model.invoke(prompt)
        return {"messages": [response]}

    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "mistral-example"}}

    query = "Hi! I'm a user interested in AI."
    language = "English"

    input_messages = [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages, "language": language},
        config,
    )
    print("\nAI Response (English):")
    output["messages"][-1].pretty_print()

    query = "What can you tell me about language models?"
    input_messages = [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages},
        config,
    )
    print("\nAI Response (continuing in English):")
    output["messages"][-1].pretty_print()

    query = "Can you switch to a different language?"
    language = "Spanish"
    input_messages = [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages, "language": language},
        config,
    )
    print("\nAI Response (Spanish):")
    output["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()

