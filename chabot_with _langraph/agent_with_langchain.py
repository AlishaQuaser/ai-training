import os
import getpass
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Please enter the Mistral API key: ")

language_model = init_chat_model("mistral-large-latest", model_provider="mistralai")
tavily_search_tool = TavilySearchResults(max_results=3)
tools = [tavily_search_tool]
language_model_with_tools = language_model.bind_tools(tools)
agent_executor = create_react_agent(language_model_with_tools, tools)
memory_handler = MemorySaver()

if __name__ == "__main__":
    print("Hello! I am your LangChain-powered agent. Type 'quit' or 'q' to exit the conversation.")

    while True:
        user_query = input("You: ")

        if user_query.lower() in ["quit", "q"]:
            print("Session ended. Goodbye!")
            break

        for response_step in agent_executor.stream(
                {"messages": [HumanMessage(content=user_query)]},
                stream_mode="values",
        ):
            response_step["messages"][-1].pretty_print()

        final_response = agent_executor.invoke({"messages": [HumanMessage(content=user_query)]})
        print(f"Agent: {final_response['messages'][-1].content}")
