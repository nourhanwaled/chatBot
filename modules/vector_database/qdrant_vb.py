from langchain.vectorstores import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from typing import List, Optional

class QdrantVectorDB:
    """Utility class for handling Qdrant vector database operations."""

    def __init__(self, collection_name: str, host: str = "localhost", port: int = 6333):
        """
        Initialize VectorDatabaseUtils.

        Args:
            collection_name: Name of the Qdrant collection
            host: Host address of the Qdrant instance
            port: Port number of the Qdrant instance
        """
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(host=host, port=port)

        # Ensure the collection exists in Qdrant
        self._initialize_collection()

    def _initialize_collection(self, vector_size: int = 768, distance: str = "Cosine"):
        """
        Ensure the Qdrant collection exists.

        Args:
            vector_size: Dimensionality of the vectors
            distance: Metric to use for similarity search (e.g., Cosine, Euclid)
        """
        collections = self.qdrant_client.get_collections().collections
        exists = any(col.name == self.collection_name for col in collections)
        
        if not exists:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance[distance.upper()])
            )

    def create_vector_store(self, text_chunks: List[str], embeddings: GoogleGenerativeAIEmbeddings) -> Qdrant:
        """
        Creates a Qdrant vector store from text chunks.

        Args:
            text_chunks: List of text chunks to embed
            embeddings: Embedding model to use

        Returns:
            Qdrant: Vector store object
        """
        # Get vector size from first embedding
        vector_size = len(embeddings.embed_query("test"))
        
        # Create collection with proper configuration
        self.qdrant_client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={
                "text_vector": {
                    "size": vector_size,
                    "distance": "Cosine"
                }
            }
        )
        
        # Create vector store with proper configuration
        vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=embeddings,
            vector_name="text_vector"
        )
        
        # Add texts with proper metadata
        texts_with_metadata = [
            {
                "text": chunk,
                "metadata": {"source": "document"}
            } for chunk in text_chunks
        ]
        
        # Add texts to vector store
        vector_store.add_texts(
            texts=[t["text"] for t in texts_with_metadata],
            metadatas=[t["metadata"] for t in texts_with_metadata]
        )
        
        return vector_store

    def load_vector_store(self, embeddings: GoogleGenerativeAIEmbeddings) -> Qdrant:
        """
        Loads the existing Qdrant vector store.

        Args:
            embeddings: Embedding model to use

        Returns:
            Qdrant: Loaded vector store object
        """
        collections = self.qdrant_client.get_collections().collections
        exists = any(col.name == self.collection_name for col in collections)
        
        if not exists:
            raise ValueError(f"Collection '{self.collection_name}' does not exist in Qdrant.")

        return Qdrant(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=embeddings,
            vector_name="text_vector"
        )

    def get_relevant_context(
        self,
        query: str,
        vector_store: Qdrant,
        k: int = 3,
        score_threshold: float = 0.7
    ) -> str:
        """
        Retrieve the most relevant context chunks for the query.

        Args:
            query: Search query
            vector_store: Vector store to search in 
            k: Number of relevant chunks to retrieve
            score_threshold: Minimum similarity score threshold (0.0 to 1.0)

        Returns:
            str: Concatenated relevant context
        """
        results = vector_store.similarity_search_with_score(query, k=k)
        
        # Filter results above threshold and extract page content
        filtered_results = [
            result.page_content 
            for result, score in results 
            if score >= score_threshold
        ]
        
        context = "\n".join(filtered_results)
        return context
