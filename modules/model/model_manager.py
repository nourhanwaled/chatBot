# model_manager.py

from modules.model.gemini_llm import Gemini
from modules.model.groq_llm import GroqLLM

class ModelManager:
    def __init__(self, google_api_key: str, groq_api_key: str):
        """
        Initialize the ModelManager with the given API key
        and register each of the supported models/variants.
        """
    

        self.models = {
            "gemini-pro":  Gemini(api_key=google_api_key, model_name="gemini-pro"),
            "gemini-1.5-pro": Gemini(api_key=google_api_key, model_name="gemini-1.5-pro"),
            "gemini-1.5-flash": Gemini(api_key=google_api_key, model_name="gemini-1.5-flash"),
            "groq-mixtral-8x7b":    GroqLLM(api_key=groq_api_key, model_name="mixtral-8x7b-32768"),
            "groq-llama-guard-3-8b": GroqLLM(api_key=groq_api_key, model_name="llama-guard-3-8b"),
            "hugging-face-distil-whisper-large-v3-en": GroqLLM(api_key=groq_api_key, model_name="distil-whisper-large-v3-en")
        }

        # Default model
        self.current_model_name = "gemini-pro"
        self.current_model = self.models[self.current_model_name]

    def change_model(self, model_name: str):
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' not recognized. "
                f"Available models: {list(self.models.keys())}"
            )
        self.current_model_name = model_name
        self.current_model = self.models[model_name]

    def get_current_model_name(self):
        return self.current_model_name

    def get_model(self):
        return self.current_model

    def generate_response(self, prompt: str) -> str:
        return self.current_model.generate_response(prompt)
