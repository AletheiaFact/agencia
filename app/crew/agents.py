from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.agents import Tool
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()
phi3_llm = Ollama(model="phi3")
chat_gpt_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
search = SerpAPIWrapper(
    params={
    	"engine": "google",
    	"gl": "br",
    	"hl": "pt",
	},
    serpapi_api_key="601e156ed7c08c1f117518eedb7f79319faa24d547538e2e16fe6254d4950918"
)
output_parser = StrOutputParser()

class Agents():
    def listQuestionsAgent():
        prompt = ChatPromptTemplate.from_messages([
			(
				"system",
				"""
				You are an expert journalist working alongside fellow agents. Your task is to identify and list crucial questions that will guide
    			your team through verifying this claim: {claim}. Each question should directly address relevant stakeholders or sources to ensure clarity and specificity.

				Your response must include a minimum of one question and no more than five questions.
    			Provide your questions in an array format and translate them only to {language}.
				"""
			),
		])

        return prompt | chat_gpt_llm | output_parser

    def researcherAgent():
        search_tool = Tool(
			name="search_tool",
			func=search.run,
			description="useful for when you need to ask with search",
		)
        prompt = ChatPromptTemplate.from_messages([
			(
				"system",
				"""
				You are an expert Reseacher Journalist with a specific role: Search content on the internet related to the claim and the provided sources
    
				Claim: {claim}
				Sources: {sources}
    			Context: {context}
    
				In case sources is provided utilize the context to gather relevant data.
				In case there is not any sources, create your query using the claim.
				
				Compiles a comprehensive response containing all relevant data from the tool
    			Provide your response in {language}.
				{agent_scratchpad}
				"""
			),
		])
        tools = [search_tool]
        agent = create_tool_calling_agent(chat_gpt_llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=False)
    
    def factCheckerAgent():
        prompt = ChatPromptTemplate.from_messages([
			(
				"system",
				"""
				You are a journalist specialized in fact-checking creates a fact-checking report relying the context gathered from the web: {messages}.
    			Use critical thinking and analytical skills to assess the accuracy and relevance of the data in relation to the following claim: {claim}.
				
    			The report should be presented in a structured JSON format. Below is the format you are to follow:
        
				{{
				"classification": "Classification from the previous user, this must be in English",
				"summary": "A concise overview reflecting your classification",
				"questions": {questions},
				"report": "A detailed narrative of your findings and evidence",
				"verification": "A detailed description of the methods and tools used for verification",
				}}

				classification: Assign one of the following labels based on your analysis:

				- Not Fact: The information lacks evidence or a factual basis.
				- Trustworthy: The information is reliable, backed by evidence or reputable sources.
				- Trustworthy, but: Generally reliable, albeit with minor inaccuracies.
				- Arguable: Subject to debate or different interpretations.
				- Misleading: Distorts the facts, causing potential misunderstandings.
				- False: Demonstrably incorrect or untrue.
				- Unsustainable: Lacks long-term viability or feasibility.
				- Exaggerated: Contains elements of truth but is overstated or embellished.
				- Unverifiable: Cannot be substantiated through reliable sources.

				- summary: Provide a succinct summary that directly supports your classification,
				offering insight into your analytical reasoning.

				- questions: Enumerate essential questions that were crucial in guiding your
				analysis and verification process, this MUST contain at least one question.

				- report: Document a comprehensive account detailing the investigative process, key findings,
				and evidence that supports your conclusion.

				- verification: Explain the specific methodologies and tools you employed to verify the
				information, highlighting your systematic approach to substantiating the claim.
    
    			compile your response in {language}, however the classification field must remain in English
				"""
			),
		])
        
        return prompt | chat_gpt_llm | output_parser
