"""Node: Generate fact-checking questions from a claim."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import AgentState

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert journalist working alongside fellow agents. Your task is to identify and list crucial questions that will guide
your team through verifying this claim: {claim}. Each question should directly address relevant stakeholders or sources to ensure clarity and specificity.

Your response must include a minimum of one question and no more than five questions.
Provide your questions in an array format and translate them only to {language}.""",
    ),
])

_chain = _prompt | llm | StrOutputParser()


def list_questions(state: AgentState) -> dict:
    result = _chain.invoke({
        "claim": state["claim"],
        "language": state["language"],
    })
    return {"questions": result}
