from langchain_google_genai import GoogleGenerativeAIEmbeddings
from credentials import API_KEY
class EmbeddingModel:
    _instance = None  # Class-level variable to store the single instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
        return cls._instance

    def __init__(self, *args):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)

    def get_embeddings(self):
        return self.embeddings

