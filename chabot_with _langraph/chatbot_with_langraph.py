from typing import Sequence
import os
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import Annotated, TypedDict

load_dotenv()


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


def create_chatbot():
    model = init_chat_model("mistral-large-latest", model_provider="mistralai")

    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

    def trim_messages(messages, max_messages=10):
        if len(messages) > max_messages:
            if isinstance(messages[0], SystemMessage):
                return [messages[0]] + messages[-(max_messages-1):]
            return messages[-max_messages:]
        return messages

    workflow = StateGraph(state_schema=State)

    def call_model(state: State):
        trimmed_messages = trim_messages(state["messages"])
        prompt = prompt_template.invoke({
            "messages": trimmed_messages,
            "language": state.get("language", "English")
        })
        response = model.invoke(prompt)
        return {"messages": [response]}

    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


def chat_session(app, thread_id="user_session_1", language="English"):
    config = {"configurable": {"thread_id": thread_id}}

    print(f"Starting chat session (ID: {thread_id}, Language: {language})")
    print("Type 'exit' to end the conversation\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        input_messages = [HumanMessage(user_input)]

        print("\nBot: ", end="")
        for chunk, _ in app.stream(
                {"messages": input_messages, "language": language},
                config,
                stream_mode="messages"
        ):
            if isinstance(chunk, AIMessage):
                print(chunk.content, end="")
        print("\n")


if __name__ == "__main__":
    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key is None:
        print("You need to set your MISTRAL_API_KEY environment variable")
        exit(1)

    chatbot = create_chatbot()
    chat_session(chatbot, thread_id="demo", language="English")
