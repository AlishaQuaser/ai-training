import json
import argparse
import logging
import os
import requests
import time
from typing import Dict, List, Union, Any
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CategoryParser:
    """
    Parse category text and generate hierarchical JSON structure using LLM assistance
    """

    def __init__(self, api_key=None, output_file="categories.json", model="mistral-large-latest"):
        """
        Initialize the parser

        Args:
            api_key (str): API key for the Mistral AI service
            output_file (str): Output file path for the JSON result
            model (str): Mistral AI model to use
        """
        self.output_file = output_file
        self.model = model

        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided. Please set MISTRAL_API_KEY environment variable or provide it as an argument.")

        self.timeout = 60
        self.max_retries = 3
        self.retry_delay = 5

    def read_file(self, file_path: str) -> str:
        """
        Read the content of a text file

        Args:
            file_path (str): Path to the text file

        Returns:
            str: Content of the file
        """
        try:
            logger.info(f"Reading file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            raise

    def chunk_text(self, text: str, max_chunk_size: int = 4000) -> List[str]:
        """
        Split text into manageable chunks for LLM processing

        Args:
            text (str): Text to split
            max_chunk_size (int): Maximum size of each chunk

        Returns:
            List[str]: List of text chunks
        """
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk)

        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks

    def parse_with_llm(self, text: str) -> Dict:
        """
        Use LLM to parse categories from the text

        Args:
            text (str): Text content containing categories

        Returns:
            Dict: Parsed categories in a hierarchical structure
        """
        logger.info("Parsing categories with Mistral AI")

        if len(text) > 4000:
            logger.info(f"Text is large ({len(text)} chars), splitting into chunks for processing")
            return self.process_large_text(text)

        prompt = f"""
        I need you to parse the following text into a hierarchical category structure.
        The text contains categories and subcategories but they don't have explicit hierarchical markers.
        
        Extract parent categories and their nested subcategories into a JSON structure.
        Each entry in the JSON should have:
        1. The parent category as the key 
        2. A list of subcategories or nested objects for further subcategories
        
        For example, if there's a parent category "Artificial Intelligence" with subcategories like 
        "Active Learning Tools", "AI Governance Tools", etc., and "Agentic AI Software" has its own 
        subcategories like "AI Agent Builders Software", the structure should reflect this hierarchy.
        
        Look for patterns that might indicate hierarchy:
        - Bold or emphasized text might indicate parent categories
        - Text with double asterisks (like **Category**) might indicate hierarchy levels
        - If you see patterns like "Category - Subcategory" or similar, interpret the hierarchy
        
        Format the output as a valid JSON object with this structure:
        {{
          "Category1": [
            "Subcategory1", 
            "Subcategory2",
            {{"Nested Subcategory1": ["Sub-sub-category1", "Sub-sub-category2"]}}
          ],
          "Category2": [...]
        }}
        
        Here's the text to parse:
        
        {text}
        
        Return only the valid JSON structure without any additional text or explanation.
        """

        return self._call_mistral_api_with_retry(prompt)

    def process_large_text(self, text: str) -> Dict:
        """
        Process large text by splitting it into chunks and combining results

        Args:
            text (str): Large text to process

        Returns:
            Dict: Combined parsed categories
        """
        chunks = self.chunk_text(text)
        all_results = {}

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")

            chunk_prompt = f"""
            I need you to parse the following text segment into a hierarchical category structure.
            This is part {i+1} of {len(chunks)} from a larger document.
            
            Extract parent categories and their nested subcategories into a JSON structure.
            Each entry in the JSON should have:
            1. The parent category as the key 
            2. A list of subcategories or nested objects for further subcategories
            
            Format the output as a valid JSON object with this structure:
            {{
              "Category1": [
                "Subcategory1", 
                "Subcategory2",
                {{"Nested Subcategory1": ["Sub-sub-category1", "Sub-sub-category2"]}}
              ],
              "Category2": [...]
            }}
            
            Here's the text segment to parse:
            
            {chunk}
            
            Return only the valid JSON structure without any additional text or explanation.
            """

            chunk_result = self._call_mistral_api_with_retry(chunk_prompt)

            all_results = self._merge_results(all_results, chunk_result)

            time.sleep(1)

        return all_results

    def _merge_results(self, dict1: Dict, dict2: Dict) -> Dict:
        """
        Merge two dictionaries of results, combining values for the same keys

        Args:
            dict1 (Dict): First dictionary
            dict2 (Dict): Second dictionary

        Returns:
            Dict: Merged dictionary
        """
        result = dict1.copy()

        for key, value in dict2.items():
            if key in result:
                if isinstance(result[key], list) and isinstance(value, list):
                    for item in value:
                        if item not in result[key]:
                            result[key].append(item)
                elif isinstance(result[key], dict) and isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            result[key].update(item)
                        else:
                            result[key][item] = []
                elif isinstance(result[key], list) and isinstance(value, dict):
                    new_dict = {k: v for item in result[key] for k, v in (item.items() if isinstance(item, dict) else {item: []})}
                    new_dict.update(value)
                    result[key] = new_dict
                elif isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._merge_results(result[key], value)
            else:
                result[key] = value

        return result

    def _call_mistral_api_with_retry(self, prompt: str) -> Dict:
        """
        Call the Mistral AI API with retry logic

        Args:
            prompt (str): Prompt for the LLM

        Returns:
            Dict: Parsed JSON response
        """
        retries = 0
        while retries <= self.max_retries:
            try:
                response = self._call_mistral_api(prompt)

                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse LLM response as JSON, attempting to extract")
                    json_str = self._extract_json(response)
                    if json_str:
                        return json.loads(json_str)
                    raise ValueError("Could not extract valid JSON from response")

            except Exception as e:
                retries += 1
                logger.warning(f"API call attempt {retries} failed: {str(e)}")

                if retries > self.max_retries:
                    logger.error(f"Maximum retries ({self.max_retries}) exceeded")
                    raise

                logger.info(f"Waiting {self.retry_delay} seconds before retry...")
                time.sleep(self.retry_delay)
                self.retry_delay *= 2

        raise RuntimeError("Failed to get response after retries")

    def _call_mistral_api(self, prompt: str) -> str:
        """
        Call the Mistral AI API to get LLM response

        Args:
            prompt (str): Prompt for the LLM

        Returns:
            str: LLM response text
        """
        if not self.api_key:
            raise ValueError("API key is required to call the Mistral AI API")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that parses category hierarchies into JSON structure."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }

        try:
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logger.error(f"API error: {response.status_code}, {response.text}")
                raise Exception(f"API error: {response.status_code}")
        except requests.exceptions.Timeout:
            logger.error(f"API request timed out after {self.timeout} seconds")
            raise
        except Exception as e:
            logger.error(f"Error calling Mistral AI API: {e}")
            raise

    def _extract_json(self, text: str) -> str:
        """
        Extract JSON string from text that might contain additional content

        Args:
            text (str): Text that might contain JSON

        Returns:
            str: Extracted JSON string or empty string if not found
        """
        import re
        matches = re.search(r'({[\s\S]*})', text)
        if matches:
            return matches.group(1)
        return ""

    def save_to_json(self, data: Dict) -> None:
        """
        Save the parsed data to a JSON file

        Args:
            data (Dict): Data to save
        """
        try:
            logger.info(f"Saving to {self.output_file}")
            with open(self.output_file, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
            logger.info(f"Successfully saved to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
            raise

    def get_input_text(self):
        """
        Get input text from user if no input file is provided

        Returns:
            str: Text input from user
        """
        print("Please enter or paste the text to parse (press Ctrl+D or Ctrl+Z+Enter when finished):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            pass
        return "\n".join(lines)

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description='Parse category text using Mistral AI and generate hierarchical JSON')
    parser.add_argument('--input', '-i', help='Path to the input text file (if not provided, will prompt for text input)')
    parser.add_argument('--output', '-o', default='categories.json', help='Path for the output JSON file')
    parser.add_argument('--api-key', '-k', help='API key for the Mistral AI service')
    parser.add_argument('--model', '-m', default='mistral-large-latest', help='Mistral AI model to use')
    parser.add_argument('--timeout', '-t', type=int, default=60, help='API request timeout in seconds')
    parser.add_argument('--retries', '-r', type=int, default=3, help='Maximum number of retries for API calls')
    parser.add_argument('--chunk-size', '-c', type=int, default=4000, help='Maximum characters per chunk when processing large texts')
    args = parser.parse_args()

    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key is None:
        logger.error("You need to set your MISTRAL_API_KEY environment variable or provide it with --api-key")
        return 1

    try:
        parser = CategoryParser(api_key=api_key, output_file=args.output, model=args.model)

        parser.timeout = args.timeout
        parser.max_retries = args.retries

        if args.input:
            text_content = parser.read_file(args.input)
        else:
            text_content = parser.get_input_text()

        if not text_content.strip():
            logger.error("No input text provided")
            return 1

        categories = parser.parse_with_llm(text_content)

        parser.save_to_json(categories)

        logger.info("Category parsing completed successfully")

    except Exception as e:
        logger.error(f"Error in main process: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

