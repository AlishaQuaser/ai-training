import os
import bs4
from typing import List
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory

conversation_history = []

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def generate(state: MessagesState):
    """Generate answer."""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    conversation_history = [
        message for message in state["messages"]
        if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
    ]
    conversation_history_str = "\n".join([msg.content for msg in conversation_history])

    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content} "
        "\n\n"
        "Conversation so far:\n"
        f"{conversation_history_str}"
    )

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
    ]

    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = llm.invoke(prompt)
    return {"messages": [response]}


if __name__ == "__main__":
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key is None:
        print("You need to set your MISTRAL_API_KEY environment variable")
        exit(1)

    model_name = "mistral-large-latest"
    model_provider = "mistralai"
    llm = init_chat_model(model_name, model_provider=model_provider)

    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_store = InMemoryVectorStore(embeddings)

    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)
    vector_store.add_documents(documents=all_splits)

    tools = ToolNode([retrieve])

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile()


    memory = {}

    agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)

    result = graph.invoke({"messages": [HumanMessage("What is Task Decomposition?")]})
    print("Question: What is Task Decomposition?")
    print(f"Answer: {result['messages'][-1].content}")

    follow_up_messages = result["messages"]
    follow_up_messages.append(HumanMessage("What are common ways of doing it?"))
    result = graph.invoke({"messages": follow_up_messages})
    print("\nQuestion: What are common ways of doing it?")
    print(f"Answer: {result['messages'][-1].content}")

    config = {"configurable": {"thread_id": "def234"}}
    input_message = (
        "What is the standard method for Task Decomposition?\n\n"
        "Once you get the answer, look up common extensions of that method."
    )
    for event in agent_executor.stream(
            {"messages": [{"role": "user", "content": input_message}]},
            stream_mode="values",
            config=config,
    ):
        event["messages"][-1].pretty_print()