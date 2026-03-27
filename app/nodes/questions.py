"""Node: Generate fact-checking questions from a claim."""

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm import get_llm
from state import AgentState

logger = logging.getLogger(__name__)

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert journalist working alongside fellow agents. Your task is to identify and list crucial questions that will guide
your team through verifying this claim: {claim}. Each question should directly address relevant stakeholders or sources to ensure clarity and specificity.

Your response must include a minimum of one question and no more than five questions.
Provide your questions in an array format and translate them only to {language}.""",
    ),
])


def list_questions(state: AgentState) -> dict:
    logger.info("[list_questions] Starting — claim='%s' language=%s", state["claim"][:80], state["language"])
    llm = get_llm()
    chain = _prompt | llm | StrOutputParser()
    result = chain.invoke({
        "claim": state["claim"],
        "language": state["language"],
    })
    logger.info("[list_questions] Generated questions (length=%d chars)", len(result))
    return {
        "questions": result,
        "reasoning_log": [
            f"[list_questions] Generated verification questions ({len(result)} chars)"
        ],
    }
