import getpass
import os
import sys
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
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

    trimmer = trim_messages(
        max_tokens=1000,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

    workflow = StateGraph(state_schema=State)

    def call_model(state: State):
        trimmed_messages = trimmer.invoke(state["messages"])
        prompt = prompt_template.invoke({
            "messages": trimmed_messages,
            "language": state["language"]
        })
        response = model.invoke(prompt)
        return {"messages": [response]}

    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "mistral-streaming-example"}}
    query = "Hi! I'm Alex. Please write a short poem about artificial intelligence."
    language = "English"

    input_messages = [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages, "language": language},
        config,
    )
    print("\nFull response (without streaming):")
    output["messages"][-1].pretty_print()

    print("\nStreaming response (token by token):")
    query = "Hi! I'm Alex. Tell me a short joke about programming."
    input_messages = [HumanMessage(query)]

    for chunk, metadata in app.stream(
            {"messages": input_messages, "language": language},
            config,
            stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage):
            print(chunk.content, end="")
            sys.stdout.flush()

    print("\n\nStreaming in a different language:")
    query = "Tell me about the stars in the night sky."
    language = "French"
    input_messages = [HumanMessage(query)]

    for chunk, metadata in app.stream(
            {"messages": input_messages, "language": language},
            config,
            stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage):
            print(chunk.content, end="")
            sys.stdout.flush()

    print("\n\nStreaming a longer response (showing progress):")
    query = "Explain how neural networks work in simple terms, with a brief example."
    language = "English"
    input_messages = [HumanMessage(query)]

    for i, (chunk, metadata) in enumerate(app.stream(
            {"messages": input_messages, "language": language},
            config,
            stream_mode="messages",
    )):
        if isinstance(chunk, AIMessage):
            print(chunk.content, end="")
            sys.stdout.flush()

            if i % 10 == 0 and i > 0:
                print(" [" + str(i) + " tokens]", end="", file=sys.stderr)
                sys.stderr.flush()


if __name__ == "__main__":
    main()

