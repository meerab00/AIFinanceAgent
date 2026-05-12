
import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

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
# TOOL: Finance Assistant
# ---------------------------
@tool
def finance_advisor(query: str) -> str:
    """Explains finance topics simply"""
    prompt = f"""
    You are a finance expert AI.

    Explain this in simple words:
    {query}
    """
    return llm.invoke(prompt).content

# ---------------------------
# TOOL: Stock Explanation
# ---------------------------
@tool
def stock_explainer(symbol: str) -> str:
    """Explains company/stock info"""
    prompt = f"""
    Explain this company stock: {symbol}
    Include business overview and investment idea.
    """
    return llm.invoke(prompt).content

tools = [finance_advisor, stock_explainer]

# ---------------------------
# SIMPLE ROUTER (NO AGENTS)
# ---------------------------
def run_finance_ai(user_input):
    if any(word.lower() in user_input.lower() for word in ["stock", "share", "company"]):
        return stock_explainer.invoke(user_input)
    else:
        return finance_advisor.invoke(user_input)

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="AI Finance Agent", page_icon="💰")

st.title("💰 AI Finance Agent (Groq + LangChain)")

user_input = st.text_input("Ask finance question:")

if st.button("Run"):
    if user_input:
        with st.spinner("Analyzing finance data..."):
            result = run_finance_ai(user_input)
        st.success(result)
    else:
        st.warning("Enter a question")
