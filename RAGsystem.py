
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

class RAGSystem:
    def __init__(self):
        print("Initializing RAG System...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.load_documents()
        
    def load_documents(self):
        """Load documents into the vector store"""
        try:
            # Load existing vector store
            if os.path.exists("vector_store_db"):
                print("Loading existing vector store...")
                self.vectorstore = FAISS.load_local(
                    "vector_store_db", 
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Added this parameter
                )
                print("Vector store loaded successfully")
            else:
                raise FileNotFoundError("Vector store not found. Please run document_processor.py first.")
                
            # Initialize the retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}  # Return top 3 most relevant documents
            )
        except Exception as e:
            print(f"Error loading documents: {e}")
            raise

    def get_relevant_context(self, query: str) -> str:
        """Get relevant context for a query"""
        print(f"\nSearching for relevant context for query: {query[:50]}...")
        try:
            # Get relevant documents
            docs = self.retriever.get_relevant_documents(query)
            
            # Combine document contents
            context = "\n\n".join(doc.page_content for doc in docs)
            
            print(f"Found {len(docs)} relevant documents")
            return context
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""
