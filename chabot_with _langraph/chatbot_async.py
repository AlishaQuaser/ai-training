import getpass
import os
import asyncio
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()


async def run_async():
    if not os.environ.get("MISTRAL_API_KEY"):
        os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")

    model = init_chat_model("mistral-large-latest", model_provider="mistralai")

    print("\n--- LangGraph Async Persistent Memory Example ---")

    async def call_model(state: MessagesState):
        response = await model.ainvoke(state["messages"])
        return {"messages": response}

    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    app = workflow.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "async123"}}

    query = "Hi! I'm Alice."
    input_messages = [HumanMessage(query)]
    output = await app.ainvoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()

    query = "What's my name?"
    input_messages = [HumanMessage(query)]
    output = await app.ainvoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()


if __name__ == "__main__":
    asyncio.run(run_async())
