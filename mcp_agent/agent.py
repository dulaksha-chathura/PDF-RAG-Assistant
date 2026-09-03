import os
import sys

# 1. Compatibility Patch for mcp / langchain-mcp-adapters import structure
import mcp.shared.context

if not hasattr(mcp.shared.context, "RequestContext"):
    mcp.shared.context.RequestContext = getattr(mcp.shared.context, "Context", None)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


async def ask_mcp_agent(user_query: str, neon_db_url: str, gemini_api_key: str):
    """
    Initializes the MCP Client, connects to the local PDF RAG server over stdio,
    retrieves available tools, and executes the ReAct Agent using Gemini.
    """
    mcp_server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")

    # Configure environment with unbuffered output to prevent stdio deadlock
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NEON_DATABASE_URL"] = neon_db_url
    env["GEMINI_API_KEY"] = gemini_api_key

    # Initialize MultiServerMCPClient
    client = MultiServerMCPClient({
        "pdf_rag_server": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-u", mcp_server_script],  # '-u' forces unbuffered stdio
            "env": env,
        }
    })

    # Retrieve MCP tools
    tools = await client.get_tools()

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        api_key=gemini_api_key,
        temperature=0
    )

    # Create ReAct agent with LLM and MCP tools
    agent = create_react_agent(llm, tools)

    # Invoke agent with user query
    response = await agent.ainvoke({"messages": [("user", user_query)]})

    # Return the final agent answer
    return response["messages"][-1].content
