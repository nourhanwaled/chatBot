from modules.model.large_language_model import LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

class GroqLLM(LLM):
    def __init__(self, api_key: str, model_name: str = "mixtral-8x7b-32768", temperature: float = 0):
        """
        Initialize the GroqLLM class with the API key and model details.

        :param api_key: The API key to authenticate with ChatGroq.
        :param model_name: The name of the model to use.
        :param temperature: The temperature setting for response generation.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.client = ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=temperature)

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using ChatGroq.

        :param prompt: The input string to generate a response for.
        :return: A generated response as a string.
        """
        try:
            system_message = "You are a helpful assistant."
            human_message = "{text}"
            chat_prompt = ChatPromptTemplate.from_messages([("system", system_message), ("human", human_message)])
            chain = chat_prompt | self.client
            response = chain.invoke({"text": prompt})
            return response.content  # Assuming the response object has a 'content' attribute
        except Exception as e:
            return f"Error generating response: {str(e)}"