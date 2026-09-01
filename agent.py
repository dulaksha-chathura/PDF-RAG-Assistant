import asyncio
import os
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_pdf_agent(user_query: str) -> str:
    """Orchestrates LangChain agent reasoning using MCP search tools."""
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Load tools exposed by the FastMCP Server
            mcp_tools = await load_mcp_tools(session)

            # Initialize Gemini model bound with MCP tools
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.0,
                google_api_key=os.getenv("GEMINI_API_KEY")
            ).bind_tools(mcp_tools)

            system_instruction = (
                "You are an expert AI document assistant. Answer user questions "
                "strictly using context retrieved from the search_pdf_documents tool."
            )

            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_query}
            ]

            # Primary Agent Execution Step
            response = await llm.ainvoke(messages)

            # Tool Execution Loop
            if response.tool_calls:
                messages.append(response)
                for tool_call in response.tool_calls:
                    selected_tool = next(t for t in mcp_tools if t.name == tool_call["name"])
                    tool_output = await selected_tool.ainvoke(tool_call["args"])
                    messages.append({
                        "role": "tool",
                        "content": str(tool_output),
                        "tool_call_id": tool_call["id"]
                    })
                
                final_response = await llm.ainvoke(messages)
                return final_response.content

            return response.content
