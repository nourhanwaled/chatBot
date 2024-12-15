# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from typing import List, Union
# from pathlib import Path
# from zipfile import ZipFile
# from io import BytesIO
# import re
# class DocUtils:

#     """Utility class for handling document processing operations."""
    
#     @staticmethod
#     def get_pdf_text(pdf_docs: List[BytesIO]) -> str:
#         """
#         Extracts text from the uploaded PDF files.
    
#         Args:
#         pdf_docs: List of BytesIO file objects
        
#         Returns:
#         str: Extracted text from all PDFs concatenated
#         """
#         text = ""
#         for pdf in pdf_docs:
#             pdf_reader = PdfReader(BytesIO(pdf.read()))
#             for page in pdf_reader.pages:
#                 extracted_text = page.extract_text()
#                 if extracted_text:
#                     text += extracted_text
#         return text

#     @staticmethod
#     def get_docx_text(docx_file):
#         """Extract text from DOCX using zipfile."""
#         text = ""
#         with ZipFile(docx_file) as docx_zip:
#             # Extract the main document XML
#             xml_content = docx_zip.read("word/document.xml").decode("utf-8")
#             # Remove all XML tags
#             cleaned_text = re.sub(r"<[^>]+>", "", xml_content)
#             text += cleaned_text
#         return text
#     @staticmethod 
#     def get_text_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
#         """
#         Splits extracted text into manageable chunks.
        
#         Args:
#             text: Text to split into chunks
#             chunk_size: Maximum size of each chunk
#             chunk_overlap: Number of characters to overlap between chunks
            
#         Returns:
#             List[str]: List of text chunks
#         """
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap
#         )
#         chunks = text_splitter.split_text(text)
#         return chunks





import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Union
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import re

# Import the Document schema from LangChain (or define your own)
from langchain.schema import Document

class DocUtils:
    """Utility class for handling document processing operations."""

    @staticmethod
    def get_pdf_documents(pdf_docs: List[BytesIO]) -> List[Document]:
        """
        Extracts pages from the uploaded PDF files and returns a list of Document objects,
        each with page content + metadata (page number, file name, etc.).

        Args:
            pdf_docs: List of BytesIO file objects

        Returns:
            List[Document]: Each Document has the text of one PDF page and metadata.
        """
        all_docs = []
        for pdf_file in pdf_docs:
            pdf_reader = PdfReader(BytesIO(pdf_file.read()))
            # Optional: If you want to preserve file_name in metadata, 
            # you might do something like pdf_file.name if it’s available.
            # For demonstration, we’ll just store a generic name or empty string:
            file_name = getattr(pdf_file, "name", "uploaded_pdf")

            for page_index, page in enumerate(pdf_reader.pages):
                extracted_text = page.extract_text()
                if extracted_text:
                    # Create a Document object, storing the page number and file name in metadata
                    doc = Document(
                        page_content=extracted_text,
                        metadata={
                            "page_number": page_index + 1,
                            "file_name": file_name
                        }
                    )
                    all_docs.append(doc)
        return all_docs

    @staticmethod
    def get_docx_documents(docx_file) -> List[Document]:
        """
        Extract text from DOCX using zipfile and return a single Document object 
        (or multiple if you prefer to parse sections).
        """
        text = ""
        with ZipFile(docx_file) as docx_zip:
            # Extract the main document XML
            xml_content = docx_zip.read("word/document.xml").decode("utf-8")
            # Remove all XML tags
            cleaned_text = re.sub(r"<[^>]+>", "", xml_content)
            text += cleaned_text

        # For consistency, let's return a list with a single Document
        # If you want to chunk or store multiple sections, you can adapt similarly
        doc = Document(
            page_content=text,
            metadata={"file_name": getattr(docx_file, "name", "uploaded_docx")}
        )
        return [doc]

    @staticmethod 
    def get_text_chunks(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """
        Splits each Document in docs into smaller chunks, preserving metadata (e.g. page_number).

        Args:
            docs: A list of Document objects, each representing text from a PDF page or doc section.
            chunk_size: Maximum size of each chunk (in characters).
            chunk_overlap: Number of characters to overlap between chunks.

        Returns:
            List[Document]: List of chunked Documents with inherited metadata.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunked_docs = []

        for doc in docs:
            # Split the page_content into chunks
            chunks = text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                # Create a new Document for each chunk, reusing metadata (including page_number)
                new_doc = Document(
                    page_content=chunk,
                    metadata=doc.metadata.copy()  # copy to avoid mutating original
                )
                chunked_docs.append(new_doc)

        return chunked_docs
