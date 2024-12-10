import streamlit as st
from PyPDF2 import PdfReader
from zipfile import ZipFile
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import time

# Import utility classes
from modules.utils.doc_utils import DocUtils
from modules.vector_database.faiss_vb import FAISSVectorDB
from modules.vector_database.qdrant_vb import QdrantVectorDB
from modules.embedding_model.embedding_model import EmbeddingModel
from modules.cache.semantic_cache import SemanticCache
from modules.cost_calculator import CostCalculator
from credentials import API_KEY

# Set page configuration
st.set_page_config(page_title="Chat with Documents")


# Utility class for handling PDF and DOCX files
class DocUtils:
    @staticmethod
    def get_pdf_text(pdf_file):
        """Extract text from PDF using PyPDF2."""
        text = ""
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
        return text

    @staticmethod
    def get_docx_text(docx_file):
        """Extract text from DOCX using zipfile."""
        text = ""
        with ZipFile(docx_file) as docx_zip:
            # Extract the main document XML
            xml_content = docx_zip.read("word/document.xml").decode("utf-8")
            # Remove all XML tags
            cleaned_text = re.sub(r"<[^>]+>", "", xml_content)
            text += cleaned_text
        return text

    @staticmethod
    def get_text_chunks(raw_text):
        """Dummy function to chunk text."""
        return [raw_text]  # Replace with actual chunking logic


class ChatPDFApp:
    def __init__(self):
        genai.configure(api_key=API_KEY)
        self.vdb_utils = QdrantVectorDB(collection_name="doc_vb", host="qdrant")
        self.semantic_cache = SemanticCache(EmbeddingModel().get_embeddings(), threshold=0.15)
        self.vector_store = self.vdb_utils.load_vector_store(EmbeddingModel().get_embeddings())
        self.model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=API_KEY)

    def user_input(self, user_question, embeddings):
        """Handles the user's input, decides whether to use cached or fresh response."""
        start_time = time.time()
        response = self.semantic_cache.ask(user_question)
        if response:
            st.write(f"Cached answer: {response}")
        else:
            context = self.vdb_utils.get_relevant_context(user_question, self.vector_store)
            prompt_template = self.get_prompt_template(context, user_question)
            response = self.model.predict(prompt_template)
            self.semantic_cache.add_to_cache(user_question, response)
            st.write("Reply (from Gemini Pro): ", response)

        end_time = time.time()
        elapsed_time = end_time - start_time
        cost, total_tokens = CostCalculator.calculate_cost("gemini-pro", user_question, response)

        st.write(f"Time taken: {elapsed_time:.2f} seconds")
        st.write(f"Total tokens: {total_tokens}")
        st.write(f"Estimated cost: ${cost:.4f}")

    def get_prompt_template(self, context, user_question):
        """Generates the appropriate prompt template based on context availability."""
        if context and context != []:
            return f"""You are a helpful and knowledgeable AI assistant. Your task is to provide accurate and comprehensive answers to questions while distinguishing whether the response is based on the provided context or your general knowledge.

                Important Instructions:
                1. If the answer is based on the provided context, clearly indicate this by starting with:
                - In English: "Based on the provided context..."
                - In Arabic: "على حسب المصادر ..."
                2. If the answer is not grounded in the provided context and relies on general knowledge, start by saying:
                - In English: "This answer is based on general knowledge..."
                - In Arabic: "الإجابة دي جاية من معلومات عامة..."
                3. Always respond in the same language as the question:
                - If the question is in English, answer entirely in English.
                - If the question is in Arabic, answer entirely in Egyptian Arabic.
                4. Be thorough and ensure clarity in your response.

            Context: {context}
            Question: {user_question}
            
            Answer: """
        else:
            return f"""You are a helpful and knowledgeable AI assistant. Your task is to provide accurate and comprehensive answers to questions while distinguishing whether the response is based on the provided context or your general knowledge.

                Important Instructions:
                1. If the answer is based on the provided context, clearly indicate this by starting with:
                - In English: "Based on the provided context..."
                - In Arabic: "على حسب المصادر ..."
                2. If the answer is not grounded in the provided context and relies on general knowledge, start by saying:
                - In English: "This answer is based on general knowledge..."
                - In Arabic: "الإجابة دي جاية من معلومات عامة..."
                3. Always respond in the same language as the question:
                - If the question is in English, answer entirely in English.
                - If the question is in Arabic, answer entirely in Egyptian Arabic.
                4. Be thorough and ensure clarity in your response.

            Question: {user_question}
            
            Answer: """
    def main(self):
        st.header("Chat with PDF && DOCX using Gemini 💬")

        if "vector_store_created" not in st.session_state:
            st.session_state["vector_store_created"] = False

        with st.sidebar:
            st.title("Upload Documents")
            uploaded_files = st.file_uploader(
                "Upload your PDF or DOCX Files", 
                accept_multiple_files=True, 
                type=["pdf", "docx"]
            )

            if st.button("Process Files"):
                if uploaded_files:
                    with st.spinner("Processing..."):
                        embeddings = EmbeddingModel().get_embeddings()
                        all_text_chunks = [] 

                        for uploaded_file in uploaded_files:
                            if uploaded_file.name.endswith(".pdf"):
                                raw_text = DocUtils.get_pdf_text(uploaded_file)
                            elif uploaded_file.name.endswith(".docx"):
                                raw_text = DocUtils.get_docx_text(uploaded_file)
                            
                            text_chunks = DocUtils.get_text_chunks(raw_text)
                            all_text_chunks.extend(text_chunks)  
                        
                        self.vdb_utils.create_vector_store(all_text_chunks, embeddings)

                        st.session_state["vector_store_created"] = True
                        st.success("Files processed and vector store updated!")
                else:
                    st.warning("Please upload files before processing.")

            if st.button("Clear Cache"):
                self.semantic_cache.clear_cache()
                st.success("Cache cleared!")

        user_question = st.text_input("Ask a question about the uploaded documents")

        if user_question and st.session_state["vector_store_created"]:
            embeddings = EmbeddingModel().get_embeddings()
            self.user_input(user_question, embeddings)
        elif user_question:
            st.warning("Please process the documents or PDFs first before asking questions.")


if __name__ == "__main__":
    app = ChatPDFApp()
    app.main()
