
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class RAGSystem:
    def __init__(self):
        print("Initializing RAG System...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Initialize an empty FAISS index
        self.vectorstore = FAISS.from_texts(
            [""], 
            embedding=self.embeddings
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        print("RAG System initialized successfully")

    def get_relevant_context(self, query: str) -> str:
        print(f"\nSearching for relevant context for query: {query[:50]}...")
        docs = self.retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        print(f"Found {len(docs)} relevant documents")
        return context

    def add_texts(self, texts: list[str]) -> None:
        print(f"\nAdding {len(texts)} new texts to vector store...")
        self.vectorstore.add_texts(texts)
        print("Texts added successfully")
