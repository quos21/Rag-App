from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pathlib import Path
import shutil
from core.rag_manager import RAGManager
from agent.tool import set_rag_instance

app = APIRouter(prefix="/rag", tags=["RAG Document Management"])
rag = RAGManager()
set_rag_instance(rag)  # Set global RAG instance

UPLOAD_DIR = Path("documents")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/documents/add")
async def add_document(
    file: UploadFile = File(...),
    doc_id: str = None
):
    """Upload and index a PDF document."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files allowed")
    
    # Save file
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Add to RAG in background
    try:
        result = rag.add_document(str(file_path), doc_id)
        return {"message": "Document added", **result}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/documents/delete/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the index."""
    try:
        result = rag.delete_document(doc_id)
        return {"message": "Document deleted", **result}
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/query")
async def query_documents(query: str = Form(...), top_k: int = Form(3)):
    """Search documents for relevant information."""
    try:
        result = rag.search(query, top_k)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/documents/get")
async def list_documents():
    """List all indexed documents."""
    docs = rag.list_documents()
    return {"total": len(docs), "documents": docs}



@app.get("/")
async def root():
    return {"status": "RAG API Running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)