from langchain_google_genai import ChatGoogleGenerativeAI
from modules.model.large_language_model import LLM

class Gemini(LLM):
    def __init__(self, api_key: str, model_name: str):
        """
        Initialize the Gemini class with the API key for ChatGoogleGenerativeAI.

        :param api_key: The API key to authenticate with ChatGoogleGenerativeAI.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client = self.initialize_client(api_key)

    def initialize_client(self, api_key: str):
        """
        Initialize the ChatGoogleGenerativeAI client.

        :param api_key: The API key to authenticate with the service.
        :return: An instance of the ChatGoogleGenerativeAI client.
        """
        # Hypothetical initialization code for ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(api_key=api_key, model=self.model_name)

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using ChatGoogleGenerativeAI.

        :param prompt: The input string to generate a response for.
        :return: A generated response as a string.
        """
        try:
            response = self.client.predict(prompt)
            return response # Assuming the response object has a 'text' attribute
        except Exception as e:
            return f"Error generating response: {str(e)}"