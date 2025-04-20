import getpass
import os
import time
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()


def main():
    if not os.environ.get("MISTRAL_API_KEY"):
        os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")

    model = init_chat_model("mistral-large-latest", model_provider="mistralai")

    print("\n--- Direct Chat Example ---")
    response1 = model.invoke([HumanMessage(content="Hi! I'm Bob")])
    print("Response 1:", response1)
    time.sleep(2)

    response2 = model.invoke([
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
        HumanMessage(content="What's my name?")
    ])
    print("Response 2:", response2)
    time.sleep(2)

    workflow = StateGraph(state_schema=MessagesState)

    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        time.sleep(2)
        return {"messages": response}

    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "abc123"}}

    print("\n--- LangGraph Persistent Memory Example ---")
    query = "Hi! I'm Bob."
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()
    time.sleep(2)

    query = "What's my name?"
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()
    time.sleep(2)

    config = {"configurable": {"thread_id": "abc234"}}
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()
    time.sleep(2)

    config = {"configurable": {"thread_id": "abc123"}}
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
