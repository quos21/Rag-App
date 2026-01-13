from fastapi import APIRouter, HTTPException, Form
from agent.decider import process_query

app=APIRouter(prefix="/bot", tags=["Chatbot"])
@app.post("/ask")
async def ask_question(
    query: str = Form(...),
    session_id: str = Form(None)
):
    """Ask a question - agent decides whether to use RAG or answer directly."""
    try:
        result = await process_query(query, session_id)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))