import asyncio
import pdb
import sys

sys.path.append(".")

from dotenv import load_dotenv

load_dotenv()


async def test_mcp_client():
    from src.utils.mcp_client import setup_mcp_client_and_tools

    test_server_config = {
        "playwright": {
            "command": "npx",
            "args": [
                "@playwright/mcp@latest",
            ],
            "transport": "stdio",
        }
    }

    mcp_tools, mcp_client = await setup_mcp_client_and_tools(test_server_config)

    pdb.set_trace()


if __name__ == '__main__':
    asyncio.run(test_mcp_client())
