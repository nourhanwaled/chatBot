from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Optional

class FAISSVectorDB:
    """Utility class for handling vector database operations."""
    
    def __init__(self, index_path: str = "faiss_index"):
        """
        Initialize VectorDatabaseUtils.
        
        Args:
            index_path: Path where the FAISS index will be stored
        """
        self.index_path = index_path
        
    def create_vector_store(self, text_chunks: List[str], embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
        """
        Creates a FAISS vector store from text chunks.
        
        Args:
            text_chunks: List of text chunks to embed
            embeddings: Embedding model to use
            
        Returns:
            FAISS: Vector store object
        """
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(self.index_path)
        return vector_store
    
    def load_vector_store(self, embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
        """
        Loads the existing FAISS vector store.
        
        Args:
            embeddings: Embedding model to use
            
        Returns:
            FAISS: Loaded vector store object
        """
        vector_store = FAISS.load_local(
            self.index_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vector_store
    
    def get_relevant_context(
        self, 
        query: str, 
        vector_store: FAISS, 
        k: int = 3
    ) -> str:
        """
        Retrieve the most relevant context chunks for the query.
        
        Args:
            query: Search query
            vector_store: Vector store to search in
            k: Number of relevant chunks to retrieve
            
        Returns:
            str: Concatenated relevant context
        """
        results = vector_store.similarity_search(query, k=k)
        context = "\n".join([result.page_content for result in results])
        return context
