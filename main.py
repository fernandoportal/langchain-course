from dotenv import load_dotenv

load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor


def main():
    result = chain.invoke(
        {
            "input": "search for 3 job postings for an ai engineer in LATAM and list their details.",
        }
    )
    print("Final Result: ", result)


if __name__ == "__main__":
    main()
