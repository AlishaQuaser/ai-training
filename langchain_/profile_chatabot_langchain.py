import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from base import load_profiles_from_db
load_dotenv()


class ChatBot:
    def __init__(self, model_name, model_provider="mistralai"):
        self.model = init_chat_model(model_name, model_provider=model_provider)
        self.profile_data = load_profiles_from_db()
        self.system_template = (
            "You are an expert recruiter. Use the following profiles to answer the user's question:\n\n{profile_data}"
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_template),
            ("user", "{question}")
        ])
        self.conversation_history = []

    def ask(self, question):
        self.conversation_history.append(HumanMessage(content=question))

        prompt = self.prompt_template.invoke({
            "profile_data": self.profile_data,
            "question": question
        })

        print("\nAnswer:")
        for token in self.model.stream(prompt):
            print(token.content, end="")
        print("\n")


if __name__ == "__main__":
    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key is None:
        print("Please set environment variable MISTRAL_API_KEY")
        exit(1)

    bot = ChatBot("mistral-large-latest")

    print("Ask your questions (type 'exit' to stop):\n")
    while True:
        user_question = input("You: ")
        if user_question.strip().lower() == "exit":
            print("Goodbye!")
            break
        bot.ask(user_question)

