from fastapi import FastAPI, UploadFile, Form
from app.model import LocalLLM
from app.loaders import load_document
from app.vectorstore import VectorDB
from app.chains import build_qa_chain


app = FastAPI(title="ChatDoc API", description="Document Q&A API using LLM", version="1.0.0")
llm = LocalLLM()
vectordb = VectorDB()
qa_chain = None


@app.get("/")
async def root():
    return {
        "message": "Welcome to ChatDoc API",
        "endpoints": {
            "/docs": "API documentation (Swagger UI)",
            "/upload": "POST - Upload a document",
            "/ask": "POST - Ask a question about uploaded documents"
        }
    }


@app.post("/upload")
async def upload_file(file: UploadFile):
    path = f"./{file.filename}"
    with open(path, 'wb') as f:
        f.write(await file.read())
    docs = load_document(path)
    vectordb.add_docs(docs)
    return {"status": "upload", "file": file.filename}

@app.post("/ask")
async def ask_question(q: str = Form(...)):
    global qa_chain
    if qa_chain is None:
        qa_chain = build_qa_chain(llm.pipe, vectordb.get_retriever())
    result = qa_chain.invoke({"query": q})
    return {"answer": result.get("result", result)}
