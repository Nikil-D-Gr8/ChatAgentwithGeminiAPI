from typing import Optional
import PyPDF2
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {e}")

def process_and_store_document(
    source_path: str, 
    output_text_path: str = "source.txt",
    vector_store_path: Optional[str] = "vector_store_db",
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> FAISS:
    """
    Process a document by extracting text, saving to file, and creating a vector store.
    
    Args:
        source_path: Path to the source PDF file
        output_text_path: Path where to save the extracted text
        vector_store_path: Path where to save the vector store (None to skip saving)
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
    
    Returns:
        FAISS vector store instance
    """
    # Extract text from PDF
    raw_text = extract_text_from_pdf(source_path)

    # Save extracted text to file
    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write(raw_text)
    print(f"Text written to {output_text_path} successfully!")

    # Create text chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    docs = text_splitter.split_documents([Document(page_content=raw_text)])

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS index
    embedding_dim = len(embeddings.embed_query("test"))
    index = faiss.IndexFlatL2(embedding_dim)

    # Create and populate vector store
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_documents(docs)

    # Save vector store if path is provided
    if vector_store_path:
        vector_store.save_local(vector_store_path)
        print(f"Vector store saved to {vector_store_path}")

    return vector_store

if __name__ == "__main__":
    # Example usage
    pdf_path = "DLUHC_How_to_rent_Oct2023.pdf"
    vector_store = process_and_store_document(pdf_path)