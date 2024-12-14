import streamlit as st
from PyPDF2 import PdfReader


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

        # Step 1: Check cache
        cached_response = self.semantic_cache.ask(user_question)
        if cached_response:
            st.write("Cached Response Found")
            moderated_response = self.moderate_output(cached_response, user_question)
            styled_response = self.style_output(moderated_response, user_question)
            st.write(f"Cached answer: {styled_response}")
            return

        # Step 2: Get context and generate raw response
        context = self.vdb_utils.get_relevant_context(user_question, self.vector_store)
        if not context:
            st.write("No relevant context retrieved!")
            return
        prompt_template = self.get_prompt_template(context, user_question)
        st.write("Generated Prompt Template:", prompt_template)

        try:
            raw_response = self.model.predict(prompt_template)
            st.write("Raw Response:", raw_response)
        except Exception as e:
            st.write("Error during raw response generation:", e)
            return

        # Step 3: Fact-checking
        fact_checked_response = self.check_facts(raw_response, context, user_question)
        st.write("Fact-Checked Response:", fact_checked_response)

        # Step 4: Moderation
        moderated_response = self.moderate_output(fact_checked_response, user_question)
        st.write("Moderated Response:", moderated_response)

        # Step 5: Styling and structuring
        styled_response = self.style_output(moderated_response, user_question)
        

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
