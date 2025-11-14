from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorDB:
    def __init__(self, persist_dir="db"):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)

    def add_docs(self, docs):
        self.db.add_documents(docs)
        self.db.persist()
    
    def get_retriever(self):
        return self.db.as_retriever(search_kwargs={"k": 3})
