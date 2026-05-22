import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

@tool
def spawn_subagent(task: str, role: str = "Expert Analyst") -> str:
    """
    Spawns a parallel sub-agent to independently research, analyze, or solve a task.
    Returns the sub-agent's final output. 
    Use this to delegate heavy thinking, complex calculations, or isolated text-generation 
    while you handle other tools.
    
    Args:
        task: The detailed task description for the sub-agent to complete.
        role: The persona or role the sub-agent should adopt (e.g. "Python Coder", "Creative Writer").
    """
    model_name = "gpt-oss:120b-cloud"  # Same as Shifu default
    base_url = "https://ollama.com/v1"
    
    # Try to grab env vars if they changed
    api_key = os.getenv("OLLAMA_API_KEY_EXECUTOR", "dummy")
    
    llm = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=0.5,
        max_tokens=4096,
    )
    
    messages = [
        SystemMessage(content=f"You are a sub-agent spawned by Shifu. Your role is: {role}.\nComplete the user's task directly, accurately, and without unnecessary preamble."),
        HumanMessage(content=f"Task: {task}")
    ]
    
    response = llm.invoke(messages)
    return response.content
