# mcp_agent/agent.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

async def ask_mcp_agent(user_query: str, neon_db_url: str, gemini_api_key: str) -> str:
    """
    Connects to the FastMCP server via Stdio transport and lets 
    the LangChain ReAct agent decide when to invoke the vector search tool.
    """
    # Initialize the LLM for the Agent reasoning loop
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=gemini_api_key
    )
    
    # Path to the MCP tool server script
    mcp_server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")

    # Connect to FastMCP server via Stdio Transport
    async with MultiServerMCPClient({
        "pdf_rag_server": {
            "command": "python",
            "args": [mcp_server_script],
            "transport": "stdio",
            "env": {
                "NEON_DATABASE_URL": neon_db_url,
                "GEMINI_API_KEY": gemini_api_key
            }
        }
    }) as mcp_client:
        
        # Get dynamic tool adapters from MCP server
        tools = await mcp_client.get_tools()
        
        # Build ReAct Agent
        agent = create_react_agent(llm, tools)
        
        # Execute Query
        result = await agent.ainvoke({"messages": [("user", user_query)]})
        
        # Return final message content
        return result["messages"][-1].content
