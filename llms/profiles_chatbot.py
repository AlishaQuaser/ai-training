import os
from mistralai import Mistral
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


class ChatBot:
    def __init__(self, _api_key, model):
        self.api_key = _api_key
        self.model = model
        self.conversation_history = []
        self.mistral_client = Mistral(api_key=api_key)
        self.initialize_context()

    def initialize_context(self):
        """
        Get the profiles data from the database
        create a string out of each row and concatenate the same into one large string
        add a new object to the conversation_history
        {
        "role": "system",
        "content": "<< large string contain details of all profiles >>"
        }

        return:
        """
        # pass
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            print("You need to set MONGO_URI in your environment variables.")
            exit(1)

        client = MongoClient(mongo_uri)
        db = client["app-dev"]
        collection = db["profiles"]

        profiles = list(collection.find())
        if not profiles:
            print("No profiles found in the database.")
            exit(1)

        all_profiles_string = ""

        for profile in profiles:
            full_name = f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip()
            area = profile.get('areaOfExpertise', 'Not specified')
            location = profile.get('currentLocation', {})
            location_str = f"{location.get('city', '')}, {location.get('state', '')}, {location.get('country', '')}"
            summary = profile.get('carrierSummary', 'No summary provided.')

            education_entries = profile.get('education', [])
            education_str = "\n".join([
                f"- {edu.get('degree')} at {edu.get('institute')} ({edu.get('startDate')})"
                for edu in education_entries
            ])

            experience_entries = profile.get('experience', [])
            experience_str = "\n".join([
                f"- {exp.get('position')} at {exp.get('company')} ({exp.get('startDate')})"
                for exp in experience_entries
            ])

            skills = profile.get('highlightedSkills', [])
            skill_str = ", ".join([s.get('name') for s in skills])

            profile_string = f"""
            Name: {full_name}
            Expertise: {area}
            Location: {location_str}
            Summary: {summary}
    
            Education:
            {education_str if education_str else 'No education data'}
    
            Experience:
            {experience_str if experience_str else 'No experience data'}
    
            Highlighted Skills:
            {skill_str if skill_str else 'No skills listed'}
            """

            all_profiles_string += profile_string.strip() + "\n\n"

        self.conversation_history.append({
            "role": "system",
            "content": f"You are provided with multiple user profiles: \n\n{all_profiles_string.strip()}"
        })

    def get_user_input(self):
        user_input = input("\nYou: ")
        user_message = {
            "role": "user",
            "content": user_input
        }
        self.conversation_history.append(user_message)
        return user_message

    def send_request(self):
        stream_response = self.mistral_client.chat.stream(
            model=self.model,
            messages=self.conversation_history,
        )

        buffer = ""
        for chunk in stream_response:
            content = chunk.data.choices[0].delta.content
            print(content, end="")
            buffer += content

        if buffer.strip():
            assistant_message = {
                "role": "assistant",
                "content": buffer
            }
            self.conversation_history.append(assistant_message)

    def run(self):
        while True:
            self.get_user_input()
            self.send_request()


if __name__ == "__main__":
    api_key = os.getenv('MISTRAL_API_KEY')
    if api_key is None:
        print('You need to set your environment variable MISTRAL_API_KEY ')
        exit(1)

    chat_bot = ChatBot(api_key, model="mistral-large-latest")
    chat_bot.run()
