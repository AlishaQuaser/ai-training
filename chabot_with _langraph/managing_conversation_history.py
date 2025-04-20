import getpass
import os
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

    config = {"configurable": {"thread_id": "mistral-conversation-example"}}

    initial_messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="Hi! My name is Alice."),
        AIMessage(content="Hello Alice! How can I assist you today?"),
        HumanMessage(content="What can you tell me about neural networks?"),
        AIMessage(content="Neural networks are computational models inspired by the human brain. They consist of layers of interconnected nodes or 'neurons' that process information. They're fundamental to modern AI systems and can learn from data to perform tasks like image recognition, language processing, and more. Is there anything specific about neural networks you'd like to know?"),
        HumanMessage(content="That's helpful, thanks."),
        AIMessage(content="You're welcome! If you have any other questions, feel free to ask."),
    ]

    query = "What is my name?"
    language = "English"

    input_messages = initial_messages + [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages, "language": language},
        config,
    )
    print("\nAI Response (name query):")
    output["messages"][-1].pretty_print()

    many_messages = initial_messages.copy()
    for i in range(10):
        many_messages.append(HumanMessage(content=f"This is message number {i+1}"))
        many_messages.append(AIMessage(content=f"I acknowledge message {i+1}"))

    query = "What did I ask you about earlier?"
    input_messages = many_messages + [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages, "language": language},
        config,
    )
    print("\nAI Response (after many messages):")
    output["messages"][-1].pretty_print()

    query = "Can you tell me what language we're speaking in now?"
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

