import os
import json
from typing import Annotated
from dotenv import load_dotenv
from openai import AzureOpenAI
from core.rag_manager import RAGManager
from core.prompts import get_rag_tool_system_prompt, get_rag_tool_user_prompt

load_dotenv()

# Global RAG instance - will be set by rag.py
_rag_instance: RAGManager = None

def set_rag_instance(rag: RAGManager):
    """Set the RAG instance to use globally"""
    global _rag_instance
    _rag_instance = rag

def get_rag_instance() -> RAGManager:
    """Get the current RAG instance"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGManager()
    return _rag_instance

# Create a separate client for the tool's LLM calls
tool_llm_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)


def answer_from_documents(
    question: Annotated[str, "The user's question to answer from company documents"]
) -> str:
    """
    Searches company documents and returns a direct answer to the question.
    Use this when the user asks about company policies, procedures, or internal information.
    Returns answer with source documents.
    """
    
    rag = get_rag_instance()
    
    # Step 1: Get relevant chunks from RAG
    search_result = rag.search(question, top_k=1)
    
    if not search_result["found"] or not search_result["results"]:
        return json.dumps({
            "answer": "I couldn't find any relevant information in the company documents to answer this question.",
            "sources": []
        })
    
    # Step 2: Extract unique sources
    sources = list(set([result['source'] for result in search_result["results"]]))
    
    # Step 3: Build context from chunks
    context = "\n\n".join([
        f"[From {result['source']}]\n{result['text']}" 
        for result in search_result["results"]
    ])
    
    # Step 4: Send to LLM to generate answer
    response = tool_llm_client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
    messages=[
        {"role": "system", "content": get_rag_tool_system_prompt()},
        {"role": "user", "content": get_rag_tool_user_prompt(context, question)}
    ],
    temperature=0.3,
    max_tokens=300
)
    
    answer = response.choices[0].message.content.strip()
    
    # Step 5: Return answer with sources as JSON
    return json.dumps({
        "answer": answer,
        "sources": sources
    })