import os
from mistralai import Mistral
import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class TextEnhancer:
    def __init__(self, api_key):
        """Initialize the TextEnhancer with Mistral client."""
        self.api_key = api_key
        self.mistral_client = Mistral(api_key=api_key)
        self.model = "mistral-large-latest"

    def enhance_text(self, raw_text):
        """
        Use Mistral client to enhance the text about an agency or freelancer.

        Args:
            raw_text (str): The raw text about the agency or freelancer

        Returns:
            str: Enhanced, more natural-sounding text
        """
        # Create a prompt that instructs Mistral how to improve the text
        prompt = f"""
        You are a professional copywriter specializing in enhancing professional profiles.
        
        Below is a raw text description of a freelancer or agency from a talent platform.
        Your task is to improve this text to make it:
        - More engaging and natural-sounding
        - Well-structured with good flow between ideas
        - Professional and confident in tone
        - Highlighting key skills and experiences effectively
        
        IMPORTANT REQUIREMENTS:
        - DO NOT summarize or shorten the text in any way
        - PRESERVE the entire content and all information from the original text
        - DO NOT remove any sections or details
        - DO NOT add any information that is not present in the original text
        - DO NOT use marketing buzzwords or hype language
        - DO NOT change any factual information, including names, locations, rates, etc.
        - DO PRESERVE all placeholder text (like "sdsdsd", "dsad") as they may be template placeholders
        - The enhanced text should be approximately the same length as the original
        
        Here is the text to enhance:
        
        {raw_text}
        
        Respond with ONLY the enhanced text, with no additional comments or explanations.
        """

        messages = [
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.mistral_client.chat(
                model=self.model,
                messages=messages
            )

            enhanced_text = response.choices[0].message.content.strip()
            return enhanced_text
        except Exception as e:
            print(f"Error calling Mistral API: {e}")
            # In case of failure, return the original text
            return raw_text


def save_text_to_file(text, filename):
    """
    Save text to a file in the current directory

    Args:
        text (str): Text to save
        filename (str): Name of the file
    """
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)
    print(f"Text saved to {filename}")


def get_raw_text():
    """Get raw text from user input."""
    print("\nEnter or paste the raw text to enhance (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if not line and lines and not lines[-1]:
            break
        lines.append(line)
    return '\n'.join(lines)


def main():
    # Check for API key
    api_key = os.getenv('MISTRAL_API_KEY')
    if api_key is None:
        print('You need to set your MISTRAL_API_KEY environment variable in the .env file')
        exit(1)

    # Get raw text from user
    raw_text = get_raw_text()

    # Exit if no text was provided
    if not raw_text.strip():
        print("No text was provided. Exiting.")
        return

    # Create output filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"enhanced_text_{timestamp}.txt"

    # Save the raw text
    raw_filename = f"raw_text_{timestamp}.txt"
    save_text_to_file(raw_text, raw_filename)

    print("\nEnhancing text using Mistral API...")

    # Enhance the text
    try:
        enhancer = TextEnhancer(api_key)
        enhanced_text = enhancer.enhance_text(raw_text)

        # Save the enhanced text
        save_text_to_file(enhanced_text, output_filename)

        print("\n=== ENHANCED TEXT ===")
        print(enhanced_text)
        print(f"\nEnhanced text saved to {output_filename}")

    except Exception as e:
        print(f"Error enhancing text: {e}")


if __name__ == "__main__":
    main()