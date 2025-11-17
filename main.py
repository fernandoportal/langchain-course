from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """Searches the web for the given query and returns a summary of the results."""
    # In a real implementation, this function would call a search API.
    print(f"Searching the web for: {query}")
    return tavily.search(query=query)


llm = ChatOpenAI(model="gpt-5")
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 jobs posting for remote work as ai engineer using langchain for South America in Linkedin?")})
    print(result)

if __name__ == "__main__":
    main()
