from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.agents import Tool

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
search = SerpAPIWrapper()
output_parser = StrOutputParser()

class CheckVerifiabilityAgent():
	def agent():
		prompt = ChatPromptTemplate.from_messages([
			(
				"system",
				"""
				You are an expert journalist with a specific role: to evaluate if the {claim} stated in {language} pertains to topics that our sources can fact-check. You are not required to verify the claim itself, just to assess its relevance to the topics provided.
				
				Topics eligible for relevance assessment include:
					- Claims related to Brazilian municipalities and Brazilian states
	
				Your response should be either 'YES' or 'NO'.
				If your answer is 'NO', please provide a detailed explanation that you dont have the correct data sources to complete the fact and why the claim does not fit the criteria for fact-checkability.
				"""
			),
		])

		return prompt | llm | output_parser

class ListQuestionsAgent():
	def agent():
		output_parser = StrOutputParser()
		
		prompt = ChatPromptTemplate.from_messages([
			(
				"system",
				"""
				You are an expert journalist working alongside fellow agents.
				
    			Your responsibility is to identify and list critical questions that will be instrumental in guiding your team through the verification process
       			for this claim: {claim}. Your response must include a minimum of one question and no more than five questions.
          		Please provide your questions in an array
				"""
			),
		])

		return prompt | llm | output_parser

#TODO: Ensure this agent is searching properly
class ResearcherAgent():
	def agent(): 
		repl_tool = Tool(
			name="python_repl",
			description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
			func=search.run,
		)
		prompt = ChatPromptTemplate.from_messages([
			(
				"system",
				"""
				You are an expert Reseacher Journalist with a specific role: Search content on the internet related to the claim and the provided sources
    
				Claim: {claim}
				Sources: {sources}
    
				If the user provided sources, analyze the source and then call the tool utilizing the sources as parameter
    			If the user did not provide any sources, create your query using the claim.
				
				Compiles a comprehensive response containing all relevant data from the tool
				{agent_scratchpad}
				"""
			),
		])
		
		tools = [repl_tool]
		agent = create_tool_calling_agent(llm, tools, prompt)
		agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
		return agent_executor
