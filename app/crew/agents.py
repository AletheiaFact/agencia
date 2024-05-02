from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
  
class CheckVerifiabilityAgent():
	def agent():
		output_parser = StrOutputParser()
		
		prompt = ChatPromptTemplate.from_messages([
			(
				"system",
				"""
				You are an expert journalist with a specific role: to evaluate if the {claim} stated in Portuguese pertains to topics that our sources can fact-check. You are not required to verify the claim itself, just to assess its relevance to the topics provided.
				
				Topics eligible for relevance assessment include:
					- Claims related to Brazilian municipalities and Brazilian states
	
				Your response should be either 'YES' or 'NO'.
				If your answer is 'NO', please provide a detailed explanation as to why the claim does not fit the criteria for fact-checkability.
				"""
			),
		])

		return prompt | llm | output_parser