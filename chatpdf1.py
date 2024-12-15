import streamlit as st
from PyPDF2 import PdfReader

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import time

from modules.utils.doc_utils import DocUtils
from modules.vector_database.faiss_vb import FAISSVectorDB
from modules.vector_database.qdrant_vb import QdrantVectorDB
from modules.embedding_model.embedding_model import EmbeddingModel
from modules.cache.semantic_cache import SemanticCache
from modules.cost_calculator import CostCalculator
from credentials import GOOGLE_API_KEY, GROQ_API_KEY  # Make sure your keys are correct
import time
from modules.model.model_manager import ModelManager  # <-- Import your ModelManager

# Set page configuration
st.set_page_config(page_title="Chat with Documents")

class ChatPDFApp:
    def __init__(self):
        # Configure Gemini (if needed, depends on your usage of generativeai)
        genai.configure(api_key=GOOGLE_API_KEY)

        # Initialize your ModelManager. 
        # If your ModelManager requires two different keys (Google vs Groq),
        # you might adapt ModelManager to accept them or store them externally.
        self.model_manager = ModelManager(google_api_key=GOOGLE_API_KEY, groq_api_key=GROQ_API_KEY)

        # Initialize your vector DB, embedding model, etc.
        self.vdb_utils = QdrantVectorDB(collection_name="doc_vb", host="qdrant")  # Or "qdrant"
        self.semantic_cache = SemanticCache(EmbeddingModel().get_embeddings(), threshold=0.15, host="qdrant")
        self.vector_store = self.vdb_utils.load_vector_store(EmbeddingModel().get_embeddings())

        # -- REMOVED self.model and self.selected_model_name
        # Because now we rely on model_manager, not a single self.model instance.

    # -- REMOVED the entire `change_model` method, because ModelManager handles switching.

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

        try:
            # Step 2B: Generate response via ModelManager
            raw_response = self.model_manager.generate_response(prompt_template)
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
        st.write("Styled Response:", styled_response)

        end_time = time.time()
        st.write(f"Processing Time: {end_time - start_time:.2f} seconds")

    def check_facts(self, response, context, user_question):
        """Verifies facts in the response against the provided context."""
        fact_check_prompt = f"""
            You are a fact-checking assistant. Your task is to verify whether the following response aligns with the context provided. If the response already aligns, confirm it. If not, highlight inaccuracies and provide corrections without altering the meaning.

            Context: {context}
            Original Response: {response}

            Fact-Checked Response:
        """
        # Replace self.model.predict with model_manager
        return self.model_manager.generate_response(fact_check_prompt)

    def moderate_output(self, response, user_question):
        """Moderates the response for tone, appropriateness, or clarity."""
        moderation_prompt = f"""
            You are a moderation assistant. Refine the response to ensure it is respectful, clear, and appropriate. Do not change the original meaning or factual accuracy.

            Original Response: {response}

            Moderated Response:
        """
        return self.model_manager.generate_response(moderation_prompt)

    def style_output(self, response, user_question):
        """Formats the response for better readability and structure."""
        style_prompt = f"""
            You are a formatting assistant. Reformat the following response to improve its readability and presentation. Use bullet points, headings, or structured paragraphs as needed. Ensure that the original meaning and content of the response is not altered.

            Original Response: {response}

            Styled Response:
        """
        return self.model_manager.generate_response(style_prompt)

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
        st.header("Chat with PDF && DOCX using LLMs 💬")

        # Maintain a flag to check if the vector store has been created
        if "vector_store_created" not in st.session_state:
            st.session_state["vector_store_created"] = False

        with st.sidebar:
            st.title("Upload Documents")
            uploaded_files = st.file_uploader(
                "Upload your PDF or DOCX Files", 
                accept_multiple_files=True, 
                type=["pdf", "docx"]
            )

            # Retrieve model options from ModelManager
            model_options = list(self.model_manager.models.keys())
            llm_model = st.selectbox("Choose a model to use:", model_options)

            if "selected_model" not in st.session_state:
                st.session_state["selected_model"] = llm_model

            if st.session_state["selected_model"] != llm_model:
                st.session_state["selected_model"] = llm_model
                self.model_manager.change_model(llm_model)  # Switch the model in the manager

                # Display a confirmation message
                message_placeholder = st.empty()
                message_placeholder.success(f"Model switched to {llm_model}!")
                time.sleep(3)
                message_placeholder.empty()
            
            # Process uploaded files
            if st.button("Process Files"):
                if uploaded_files:
                    with st.spinner("Processing..."):
                        embeddings = EmbeddingModel().get_embeddings()
                        all_text_chunks = []

                        for uploaded_file in uploaded_files:
                            if uploaded_file.name.endswith(".pdf"):
                                raw_text = DocUtils.get_pdf_text([uploaded_file])
                            elif uploaded_file.name.endswith(".docx"):
                                raw_text = DocUtils.get_docx_text(uploaded_file)
                            
                            text_chunks = DocUtils.get_text_chunks(raw_text)
                            all_text_chunks.extend(text_chunks)  

                        self.vdb_utils.create_vector_store(all_text_chunks, embeddings)
                        
                        st.session_state["vector_store_created"] = True
                        st.success("Files processed and vector store updated!")
                else:
                    st.warning("Please upload files before processing.")

            # Clear cache
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
