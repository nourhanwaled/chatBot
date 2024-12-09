import uuid
import time
from typing import List
# from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, SearchParams
from qdrant_client.http import models

class SemanticCache:
    def __init__(self, embedding_model, threshold=0.35):
        self.encoder = embedding_model
        self.cache_client =  QdrantClient(host="192.168.1.110", port=6333) 
        self.cache_collection_name = "caching"

        if not self.collection_exists(self.cache_collection_name):
            print(f"Collection '{self.cache_collection_name}' does not exist. Creating a new one.")
            self.cache_client.create_collection(
                collection_name=self.cache_collection_name,
                vectors_config=models.VectorParams(
                    size=len(embedding_model.embed_query("Hello")),
                    distance='Euclid'
                )
            )
        else:
            print(f"Collection '{self.cache_collection_name}' already exists. Loading it.")


        
        self.euclidean_threshold = threshold
    
    
    def collection_exists(self, collection_name: str) -> bool:
        try:
            self.cache_client.get_collection(collection_name)
            return True
        except Exception as e:
            return False

    def get_embedding(self, question):
        print("Embedding question: ", question)
        embedding = self.encoder.embed_query(question)
        return embedding

    def search_cache(self, embedding):
        search_result = self.cache_client.search(
            collection_name=self.cache_collection_name,
            query_vector=embedding,
            limit=1
        )
        return search_result

    def add_to_cache(self, question, response_text):
        point_id = str(uuid.uuid4())
        vector = self.get_embedding(question)
        point = PointStruct(id=point_id, vector=vector, payload={"response_text": response_text})
        self.cache_client.upload_points(
            collection_name=self.cache_collection_name,
            points=[point]
        )
        
    def ask(self, question):
        vector = self.get_embedding(question)
        # print("Asking cache: ",vector[30:35])
        search_result = self.search_cache(vector)
        # print("Search_Result: ", search_result)
        if search_result:
            for s in search_result:
                # print("S.Score: ", s.score)
                if s.score <= self.euclidean_threshold:
                    print('Answer recovered from Cache.')
                    print(f'Found cache with score {s.score:.3f}')
                  
            
                    return s.payload['response_text']

        print('No answer found in Cache.')
        return None