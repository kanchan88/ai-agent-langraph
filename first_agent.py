import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# creating agent memory 
class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str

load_dotenv()

# temp = 0 focused and deterministic response | 1= creative | 2 = wild
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# adding agent capabilities

def classification_node(state: State):
    """
    Classify the text into predefined categories [News, Research, Blog, Other]

    Parameter:
    state (State): The current state dictionary containing the text to classify

    Return:
    A dictionary with the "classification" key containing the category result

    """

    # prompt template to ask model classify the given text
    prompt = PromptTemplate(
        input_variables= ["text"],
        template= 
        """
            Classify the following text into one of the categories: News, Research, Blog, Others.

            Text: {text}

            Category:
        """
    )

    # Format the prompt with the input text from the state
    message = HumanMessage(content=prompt.format(text=state["text"]))

    # invoke language model to classify text
    classification = llm.invoke([message]).content.strip()

    # return classification in dict
    return {"classification":classification}


def entity_extraction_node(state: State):
    # Function to identify and extract named entities from text
    # Organized by category (Person, Organization, Location)
    
    # Create template for entity extraction prompt
    # Specifies what entities to look for and format (comma-separated)
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText:{text}\n\nEntities:"
    )
    
    # Format the prompt with text from state and wrap in HumanMessage
    message = HumanMessage(content=prompt.format(text=state["text"]))
    
    # Send to language model, get response, clean whitespace, split into list
    entities = llm.invoke([message]).content.strip().split(", ")
    
    # Return dictionary with entities list to be merged into agent state
    return {"entities": entities}


def summarization_node(state):
    """
    Summarize the given text using models

    Parameter:
    state: takes text input from state

    Returns:
    summary text in dict form key: summary
    """

    # ask model to summarize input text in one sentence
    summarization_prompt = PromptTemplate.from_template(
        """Summarize the following text in one short sentence.
        
        Text: {input}
        
        Summary:"""
    )

    # The "|" operator pipes the output of the prompt into the model
    chain = summarization_prompt | llm

    response = chain.invoke({"input": state["text"]})

    return {"summary": response.content}


# the Agent Structure
workflow = StateGraph(State)

# add nodes to the graph
workflow.add_node("classification", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarization_node)

# add edges
workflow.set_entry_point("classification")
workflow.add_edge("classification", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)

# compile the graph
app  = workflow.compile()

# Define a sample text about Anthropic's MCP to test our agent
sample_text = """
Anthropic's MCP (Model Context Protocol) is an open-source powerhouse that lets your applications interact effortlessly with APIs across various systems.
"""

# Create the initial state with our sample text
state_input = {"text": sample_text}

# Run the agent's full workflow on our sample text
result = app.invoke(state_input)
print(result)

# Print each component of the result:
# - The classification category (News, Blog, Research, or Other)
print("Classification:", result["classification"])

# - The extracted entities (People, Organizations, Locations)
print("\nEntities:", result["entities"])

# - The generated summary of the text
print("\nSummary:", result["summary"])