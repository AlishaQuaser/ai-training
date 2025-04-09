# import os
# from mistralai import Mistral
#
#
# class ChatBot:
#     def __init__(self, api_key, model):
#         self.api_key = api_key
#         self.model = model
#         self.client = Mistral(api_key=api_key)
#         self.conversation_history = []
#
#     def run(self):
#         while True:
#             try:
#                 self.get_user_input()
#                 self.send_request()
#             except KeyboardInterrupt:
#                 print("\nExiting...")
#                 break
#
#     def get_user_input(self):
#         user_input = input("\nYou: ")
#         user_message = {
#             "role":"user",
#             "content":user_input
#         }
#         self.conversation_history.append(user_message)
#
#     def send_request(self):
#         print("Assistant: ", end="", flush=True)
#         response = self.client.chat.stream(
#             model=self.model,
#             messages=self.conversation_history,
#         )
#
#         message_content = ""
#         for chunk in response:
#             if chunk.choices:
#                 delta = chunk.choices[0].delta
#                 if delta and delta.content:
#                     print(delta.content, end="", flush=True)
#                     message_content += delta.content
#
#         # Add assistant's full message to history
#         self.conversation_history.append(assistant_message)
#         print()  # For newline after assistant's response
#
#
# if __name__ == "__main__":
#     api_key = os.getenv('MISTRAL_API_KEY')
#     if api_key is None:
#         print('You need to set your MISTRAL_API_KEY environment variable')
#         exit(1)
#
#     chat_bot = ChatBot(api_key=api_key, model="mistral-large-latest")
#     chat_bot.run()
