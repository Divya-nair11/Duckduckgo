import asyncio
import sys
import os
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from functools import wraps

class BedrockKBMCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.agent: Optional[AgentExecutor] = None
        self.llm = None
        self.tools = []
        self.last_web_search_time = 0
        self.web_search_lock = asyncio.Lock()

        self.aws_config = {
            "AWS_ACCESS_KEY_ID": "",
            "AWS_SECRET_ACCESS_KEY": "",
            "AWS_REGION": "us-east-2",
            "KNOWLEDGE_BASE_ID": ""
        }

    def rate_limit_web_search(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with self.web_search_lock:
                current_time = asyncio.get_event_loop().time()
                time_since_last_search = current_time - self.last_web_search_time
                if time_since_last_search < 1.5:
                    await asyncio.sleep(1.5 - time_since_last_search)
                result = await func(*args, **kwargs)
                self.last_web_search_time = asyncio.get_event_loop().time()
                return result
        return wrapper

    async def connect_to_kb_server(self):
        env = os.environ.copy()
        env.update(self.aws_config)

        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-aws-kb-retrieval"],
            env=env
        )

        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.session = await self.exit_stack.enter_async_context(ClientSession(*stdio_transport))
            await self.session.initialize()

            kb_tools = await load_mcp_tools(self.session)
            for tool in kb_tools:
                if tool.name == "retrieve_from_aws_kb":
                    tool.args_schema["knowledgeBaseId"] = self.aws_config["KNOWLEDGE_BASE_ID"]
            self.tools.extend(kb_tools)

        except Exception as e:
            print(f"Failed to connect to AWS KB server: {str(e)}")
            raise

    async def connect_to_web_search_server(self):
        web_search_path = os.path.abspath("C:/Users/divya.p/Documents/web_search/duckduckgo-mcp-server/build/index.js")

        server_params = StdioServerParameters(
            command="node",
            args=[web_search_path]
        )

        try:
            await asyncio.sleep(1.5)
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            web_session = await self.exit_stack.enter_async_context(ClientSession(*stdio_transport))
            await web_session.initialize()

            web_tools = await load_mcp_tools(web_session)
            for tool in web_tools:
                if tool.name == "duckduckgo_web_search":
                    tool.func = self.rate_limit_web_search(tool.func)
            self.tools.extend(web_tools)

        except Exception as e:
            print(f"Failed to connect to DuckDuckGo Search server: {str(e)}")
            raise

    def _setup_bedrock_agent(self, tools):
        self.llm = ChatBedrock(
            provider="anthropic",
            model_id="arn:aws:bedrock:us-east-2:528757829695:inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:0",
            temperature=0,
            max_tokens=3000,
            region_name=self.aws_config["AWS_REGION"],
            aws_access_key_id=self.aws_config["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=self.aws_config["AWS_SECRET_ACCESS_KEY"],
        )

        sorted_tools = sorted(
            tools,
            key=lambda t: 0 if "retrieve_from_aws_kb" in t.name else 1
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant with access to both a company knowledge base and the web.

Follow this strict decision order:
1. Use `retrieve_from_aws_kb` to retrieve relevant information from the KB.
   - If the KB provides a valid, non-empty response (i.e., not "No relevant information found" or similar), use that information to answer the query and stop. Do not proceed to web search.
2. Only if the KB has no relevant data (i.e., returns "No relevant information found" or similar), then use `duckduckgo_web_search` to answer the query.

Clearly label the source in your response:
- [KNOWLEDGE BASE]: info from KB
- [WEB SEARCH]: from internet
- [GENERAL KNOWLEDGE]: if based on your internal understanding
- [COMBINED]: if multiple used
"""),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        agent = create_tool_calling_agent(self.llm, sorted_tools, prompt)
        self.agent = AgentExecutor(agent=agent, tools=sorted_tools, verbose=True, max_iterations=2)

    async def process_query(self, query: str) -> str:
        if not self.agent or not self.session:
            raise RuntimeError("Agent not initialized. Call connect methods first.")

        try:
            result = await self.agent.ainvoke({"input": query})
            for step in result.get("intermediate_steps", []):
                action, observation = step
                if action.tool == "retrieve_from_aws_kb":
                    if observation and isinstance(observation, list) and len(observation) > 0:
                        kb_response = observation[0].get("snippet", "")
                        if kb_response and "No relevant information found" not in kb_response:
                            return f"Answer: [KNOWLEDGE BASE] {kb_response}"

            output = result.get("output", "No response generated")
            if isinstance(output, list):
                output = " ".join(str(item) for item in output)
            return f"Answer: {output}"
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            try:
                messages = [
                    SystemMessage(content="Answer using general knowledge. KB and web search failed."),
                    HumanMessage(content=query)
                ]
                llm_response = await self.llm.ainvoke(messages)
                return f"[Fallback Response] {llm_response.content}"
            except Exception as fallback_error:
                return f"Error: Unable to process query. {str(fallback_error)}"

    async def test_kb_connection(self):
        if not self.session:
            print("No session established")
            return False

        try:
            print(f"Available tools: {[tool.name for tool in self.tools]}")
            return True
        except Exception as e:
            print(f"KB connection test failed: {str(e)}")
            return False

    async def chat_loop(self):
        print("Type 'quit' to exit, 'test' to test connection\n")
        while True:
            try:
                query = input("You: ").strip()
                if query.lower() == 'quit':
                    print("Goodbye!")
                    break
                elif query.lower() == 'test':
                    await self.test_kb_connection()
                    continue
                elif not query:
                    continue
                print("Processing...")
                response = await self.process_query(query)
                print(f"Assistant: {response}\n")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

    async def cleanup(self):
        try:
            if self.session:
                await self.exit_stack.aclose()
        except Exception as e:
            print(f"Cleanup warning: {str(e)}")
        finally:
            await asyncio.sleep(0.1)

async def main():
    client = BedrockKBMCPClient()
    try:
        print("Connecting to MCP servers...")
        await client.connect_to_kb_server()
        await client.connect_to_web_search_server()
        client._setup_bedrock_agent(client.tools)
        connection_ok = await client.test_kb_connection()
        if connection_ok:
            await client.chat_loop()
        else:
            print("Failed to connect to servers.")
    except Exception as e:
        print(f"Client failed: {str(e)}")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
