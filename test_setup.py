import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# load environment variable
load_dotenv()

# initialize ChatOpenAi instance
llm = ChatOpenAI(model="gpt-4o-mini")

# test setup
response = llm.invoke("I am building AI agent. What do you think? Respond me in 3 words!")
print(response.content)