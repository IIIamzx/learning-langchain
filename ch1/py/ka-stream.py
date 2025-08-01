from langchain_core.runnables import chain
from langchain_core.prompts import ChatPromptTemplate
from llm_factory import LLMFactory

model = LLMFactory.get_llm("qwen-plus")


template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)


@chain
def chatbot(values):
    prompt = template.invoke(values)
    for token in model.stream(prompt):
        yield token


for part in chatbot.stream({"question": "where is capital of China ? "}):
    print(part)
