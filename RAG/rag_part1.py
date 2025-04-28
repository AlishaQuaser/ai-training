import getpass
import os
import bs4
from typing import List
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


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

    prompt = hub.pull("rlm/rag-prompt")
    example_messages = prompt.invoke(
        {"context": "(context goes here)", "question": "(question goes here)"}
    ).to_messages()

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    result = graph.invoke({"question": "What is Task Decomposition?"})
    print(f'Answer: {result["answer"]}')