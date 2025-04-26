import os
import asyncio
import base64
import pdb
from typing import List, Tuple, Optional
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
import base64
import json
import logging
from typing import Optional, Dict, Any, Type
from langchain_core.tools import BaseTool
from pydantic.v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


async def setup_mcp_client_and_tools(mcp_server_config: Dict[str, Any]) -> Tuple[
    Optional[List[BaseTool]], Optional[MultiServerMCPClient]]:
    """
    Initializes the MultiServerMCPClient, connects to servers, fetches tools,
    filters them, and returns a flat list of usable tools and the client instance.

    Returns:
        A tuple containing:
        - list[BaseTool]: The filtered list of usable LangChain tools.
        - MultiServerMCPClient | None: The initialized and started client instance, or None on failure.
    """

    logger.info("Initializing MultiServerMCPClient...")

    try:
        client = MultiServerMCPClient(mcp_server_config)
        await client.__aenter__()
        mcp_tools = client.get_tools()
        logger.info(f"Total usable MCP tools collected: {len(mcp_tools)}")
        return mcp_tools, client

    except Exception as e:
        logger.error(f"Failed to setup MCP client or fetch tools: {e}", exc_info=True)
        return [], None
