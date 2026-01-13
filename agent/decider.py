import os
import json
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from .tool import answer_from_documents
from core.prompts import get_agent_system_message, get_agent_task_prompt

load_dotenv()

session_memory: Dict[str, list] = {}


def get_azure_model_client(temperature: float = 0.3):
    return AzureOpenAIChatCompletionClient(
        model=os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=temperature,
    )


agent = AssistantAgent(
    name="Assistant",
    model_client=get_azure_model_client(),
    tools=[answer_from_documents],
    system_message=get_agent_system_message(),
)


async def process_query(query: str, session_id: str | None = None) -> Dict[str, Any]:
    if session_id:
        session_memory.setdefault(session_id, [])

    context = ""
    if session_id and session_memory[session_id]:
        context = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in session_memory[session_id][-6:]
        )

    task = get_agent_task_prompt(context,query)

    result = await agent.run(task=task)

    answer = ""
    sources_used = []
    
    # Check if tool was called and extract sources from tool output
    for msg in result.messages:
        if hasattr(msg, 'content') and isinstance(msg.content, list):
            for item in msg.content:
                if hasattr(item, 'content') and isinstance(item.content, str):
                    try:
                        tool_result = json.loads(item.content)
                        if isinstance(tool_result, dict) and 'sources' in tool_result:
                            sources_used = tool_result['sources']
                            # Also extract the actual answer from tool output
                            if 'answer' in tool_result:
                                answer = tool_result['answer']
                    except:
                        pass
    
    # If no answer extracted from tool, get agent's final response
    if not answer:
        for msg in reversed(result.messages):
            if getattr(msg, "source", None) == "Assistant":
                answer = str(msg.content)
                break

    if session_id:
        session_memory[session_id].extend(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer},
            ]
        )
        session_memory[session_id] = session_memory[session_id][-20:]

    return {
        "answer": answer,
        "sources": sources_used if sources_used else None
    }


if __name__ == "__main__":
    async def test():
        print("Test 1:")
        print(await process_query("What is Python?", "s1"))
        print("\nTest 2:")
        print(await process_query("What is the recruitment policy?", "s1"))
        print("\nTest 3:")
        print(await process_query("Summarize that in one line", "s1"))

    asyncio.run(test())