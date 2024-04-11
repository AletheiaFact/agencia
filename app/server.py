#!/usr/bin/env python
import re
import json
import operator
import functools
import os
os.environ["SERPER_API_KEY"] = "serper_api_key"

from langchain.agents import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from fastapi import FastAPI
from langserve import add_routes

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
)
from langchain.tools.render import format_tool_to_openai_function
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

from typing import Annotated, Sequence, TypedDict
from typing_extensions import TypedDict

## Define Large Language Model (LLM)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key="open_api_key"
)

## Define tools
search = GoogleSerperAPIWrapper()
search_tool = Tool(
    name="search_tool",
    func=search.run,
    description="Useful for when you need to ask with search on internet",
)

tools = [search_tool]

## Create Agents
def create_agent(llm, tools, system_message: str, action_description: str):
    """Create an agent with a given LLM, tools, system message, and action description."""
    functions = [format_tool_to_openai_function(t) for t in tools] if tools else []

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a journalist specialized in fact-checking, collaborating with other assistants.
                Use the provided tools to progress towards answering the question.
                To structure your findings and actions, you will utilize a specific JSON template.
                This template serves as a standardized format for documenting your investigative process,
                decisions made, and the information verified.
                Below is the JSON template you are to use and update throughout your interactions:
                ```
                {{
                    "can_be_fact_checked": "boolean value indicating if the claim can be checked",
                    "step_description": "a brief description of your action or an explanation based on {action_description}",
                    "isFinalAnswer": "boolean value indicating if this is the final answer",
                    "response": "detailed response or findings from your investigation"
                }}
                ```
                Critical Instructions for Your Interactions:
                - Continuous Update: It is imperative to continuously update the JSON template based on the inputs and findings from previous assistants. This iterative process is crucial for building upon the collective knowledge and insights gathered by the team.
                - Each iteration must have a different step_description from the previous one.
                - Final Answer Protocol: Should you or any other assistant determine the final answer or a definitive deliverable:
                    - If you have the final answer, set the isFinalAnswer field in the JSON template to true. This signals to the entire team that the investigation has concluded, and the question at hand has been resolved.
                    - Conversely, if the investigation is ongoing and the final answer remains elusive, set the isFinalAnswer field to false. This indicates that further analysis and investigation are required.
                
                Available Tools: {tool_names}.

                Additional Instructions: {system_message}
                """
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message, action_description=action_description)

    if functions:
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        return prompt | llm.bind_functions(functions)
    else:
        # If there are no functions, set tool_names to 'None'.
        prompt = prompt.partial(tool_names="None")
        return prompt | llm

## Create graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

## Define Agent Nodes
# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    if isinstance(result, FunctionMessage):
        pass
    else:
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        "sender": name,
    }

supervisor_agent = create_agent(
    llm,
    [],
    system_message="""
        The primary objective is to verify if the provided claim can be fact-checked.
        Based on your evaluation, you will either proceed the task with next agent
        or clarify why the claim cannot be verified. Follow the structured guidelines below depending
        on the feasibility of fact-checking the claim:

        Claims that can be fact-checked: Claims which talk about education or can be substantiated through web searches

        For Verifiable Claims:
        If you encounter a claim related to education or information that can be verified through web searches, follow these steps:

        - "can_be_fact_checked": Set this field to true, indicating the claim is open to verification.
        - "isFinalAnswer": Set this to false, as the claim's verification process is underway but not yet concluded.
        - Proceeding: Hand over the investigation to the next agent for further action or analysis.
        
        For Unverifiable Claims:
        Should the claim be beyond the scope of fact-checking due to its nature or lack of available information, adhere to the following protocol:

        "isFinalAnswer": Set this to true, signifying that no further action can lead to the claim's verification.
        "can_be_fact_checked": Set this to false, reflecting that the claim cannot be substantiated.
        "response": Provide a comprehensive explanation detailing why the claim cannot be verified. This might include the absence of reliable sources, the speculative nature of the claim, or any other pertinent factor.
    """,
    action_description="Verifing if it can be fact-checked."
)
supervisor_node = functools.partial(agent_node, agent=supervisor_agent, name="Supervisor")

data_researcher_agent = create_agent(
    llm,
    [search_tool],
    system_message="""
        As the Data Researcher, your mission is to deepen our understanding of the claim by gathering more precise and relevant data.
        
        Here's how you can make a difference:
            - Utilize the provided tools, such as search engines and databases, to extract additional information that could shed light on the claim.
            - Emphasize accuracy and relevance in the data you collect.
            - Expand your search beyond the initial data points, exploring related topics or queries that could provide further context or evidence.
            - Engage in critical evaluation of the sources you encounter. Consider their reliability and the credibility of the information they offer.
            - Ensure 'isFinalAnswer' is always set to 'False' as your role involves preliminary data gathering and analysis, not concluding the investigation.
            - "isFinalAnswer": You must set this to False,

        **Special Instruction for Using Tools:**
            If you use a tool and obtain a result, immediately integrate its output into the 'response' field of your JSON object. This should be part of your overall findings. Avoid creating separate responses for the tool's output; instead, merge it with your analysis in a cohesive manner within the single JSON structure provided below. 

            Your consolidated findings should adhere to this structure:
            ```json
            {
                "can_be_fact_checked": true,
                "step_description": "Your investigative process, including any tool use.",
                "isFinalAnswer": You must set this to False,
                "response": "Your detailed findings, inclusive of any tool outputs, providing a comprehensive view of the claim."
            }
            ```
    """,
    action_description="Collecting necessary data."
)
data_researcher_node = functools.partial(agent_node, agent=data_researcher_agent, name="Data Researcher")

fact_checker_agent = create_agent(
    llm,
    [],
    system_message="""
        Upon receiving data from the Data Researcher, your task involves a deep and thorough analysis
        to derive an informed conclusion. The culmination of your efforts should be presented in a
        structured JSON format. However, in this instance, all elements of your analysis - including
        classification, summary, questions, report, and verification steps - must be nested within the
        "response" field of your JSON object. This structured approach ensures clarity and coherence
        in presenting your findings. Below is the format you are to follow:
        
        {{
            "can_be_fact_checked": Boolean value indicating if the claim can be checked,
            "step_description": A brief description of your action: Creating Fact-checking Report,
            "isFinalAnswer": Indicate whether your findings conclusively address the query. A value of true suggests that no further investigation is necessary, marking the end of the inquiry. Conversely, a value of false signals the need for additional investigation,
            "response": {
                "classification": "Your analysis-based categorization",
                "summary": "A concise overview reflecting your classification",
                "questions": ["Key inquiries addressed during your verification"],
                "report": "A detailed narrative of your findings and evidence",
                "verification": "A detailed description of the methods and tools used for verification"
            }
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
    """,
    action_description="Writing the Fact-checking Report."
)
fact_checker_node = functools.partial(agent_node, agent=fact_checker_agent, name="Fact checker")

tool_executor = ToolExecutor(tools)

def tool_node(state):
    """This runs tools in the graph
    It takes in an agent action and calls that tool and returns the result."""
    messages = state["messages"]
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    tool_input = json.loads(
        last_message.additional_kwargs["function_call"]["arguments"]
    )
    # We can pass single-arg inputs by value
    if len(tool_input) == 1 and "__arg1" in tool_input:
        tool_input = next(iter(tool_input.values()))
    tool_name = last_message.additional_kwargs["function_call"]["name"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input=tool_input,
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    content = f"{tool_name} response: {str(response)}\n"
    function_message = FunctionMessage(content=content, name=action.tool)
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}


def extract_first_object(text):
    pattern = r'\{[^\{]*?\}'
    match = re.search(pattern, text)
    
    if match:
        return match.group(0)

# Define Edge Logic
# We can define some of the edge logic that is needed to decide what to do based on results of the agents
# Either agent can decide to end
def router(state):
    messages = state["messages"]
    last_message = messages[-1]
    last_message_json = ""

    try:
        last_message_json = json.loads(last_message.content)
    except:
        last_message.content = extract_first_object(last_message.content)
        last_message_json = json.loads(last_message.content)
        
    if "function_call" in last_message.additional_kwargs:
        return "call_tool"
    if last_message_json["isFinalAnswer"] == True:
        return "end"
    return "continue"


# Define the Graph
# We can now put it all together and define the graph!
workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Data Researcher", data_researcher_node)
workflow.add_node("Fact checker", fact_checker_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "Supervisor",
    router,
    {"continue": "Data Researcher", "call_tool": "call_tool", "end": END},
)

workflow.add_conditional_edges(
    "Data Researcher",
    router,
    {"continue": "Fact checker", "call_tool": "call_tool", "end": "Fact checker"},
)

workflow.add_conditional_edges(
    "Fact checker",
    router,
    {"continue": END, "call_tool": "call_tool", "end": END},
)

workflow.add_conditional_edges(
    "call_tool",
    # Each agent node updates the 'sender' field
    # the tool calling node does not, meaning
    # this edge will route back to the original agent
    # who invoked the tool
    lambda x: x["sender"],
    {
        "Supervisor": "Supervisor",
        "Data Researcher": "Data Researcher",
        "Fact checker": "Fact checker",
    },
)
workflow.set_entry_point("Supervisor")
agents_app = workflow.compile()

app = FastAPI(
    title="Aletheia Server",
    version="1.1",
    description="Spin up aletheia api server using LangChain's Runnable interfaces",
)

add_routes(
    app,
    agents_app
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)