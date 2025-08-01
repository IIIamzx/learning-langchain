from llm_factory import LLMFactory
from langchain_core.prompts import ChatPromptTemplate

# the building blocks

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)

model = LLMFactory.get_llm("qwen-plus")

# combine them with the | operator

chatbot = template | model

# use it

response = chatbot.invoke({"question": "你是谁?"})
print(response.content)

# streaming

for part in chatbot.stream({"question": "你是谁?"}):
    print(part)
