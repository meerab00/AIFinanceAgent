import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# ---------------------------
# API KEY
# ---------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY")
    st.stop()

# ---------------------------
# LLM
# ---------------------------
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant"
)

# ---------------------------
# TOOL 1: Simple Finance Helper
# ---------------------------
@tool
def finance_advisor(query: str) -> str:
    """
    Gives basic financial guidance and explanations.
    """
    prompt = f"""
    You are a financial assistant.
    Explain in simple words:

    {query}
    """
    return llm.invoke(prompt).content

# ---------------------------
# TOOL 2: Stock Info (simple demo tool)
# ---------------------------
@tool
def stock_info(symbol: str) -> str:
    """
    Explain about a stock symbol (no real-time API in this basic version).
    """
    prompt = f"""
    Explain about stock {symbol}.
    Include what company does and basic investment idea.
    """
    return llm.invoke(prompt).content

tools = [finance_advisor, stock_info]

# ---------------------------
# AGENT
# ---------------------------
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("💰 AI Finance Agent (LangChain + Groq)")

user_input = st.text_input("Ask finance question:")

if st.button("Run Agent"):
    if user_input:
        result = agent_executor.invoke({"input": user_input})
        st.write(result["output"])
    else:
        st.warning("Enter a question")
