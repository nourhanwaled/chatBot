from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Union
from pathlib import Path

class DocUtils:
    """Utility class for handling document processing operations."""
    
    @staticmethod
    def get_pdf_text(pdf_docs: Union[List[str], List[Path]]) -> str:
        """
        Extracts text from the uploaded PDF files.
        
        Args:
            pdf_docs: List of PDF file paths or file objects
            
        Returns:
            str: Extracted text from all PDFs concatenated
        """
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
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
    def get_text_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """
        Splits extracted text into manageable chunks.
        
        Args:
            text: Text to split into chunks
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List[str]: List of text chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(text)
        return chunks
