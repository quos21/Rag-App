from fastapi import FastAPI
from apis.rag import app as rag_router
from apis.query import app as bot_router
from middleware import XAuthMiddleware
from fastapi.middleware.cors import CORSMiddleware
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(title="RAG Chatbot API")

protected_paths = ["/rag", "/bot"]
app.add_middleware(
    XAuthMiddleware,
    protected_paths=protected_paths,
    env_var_name="X_AUTH_TOKEN",
)


app.include_router(rag_router)
app.include_router(bot_router)

@app.get("/")
async def root():
    return {
        "message": "RAG Chatbot API",
        "endpoints": {
            "rag": "/rag",
            "bot": "/bot",
            "docs": "/docs"
        }
    }